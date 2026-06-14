"""CRE-specific case configuration.

Base case parameters (date, time ranges, atmospheric levels, cloud
microphysics, clear/cloudy flag) live in the shared
``lrt_sim.ssfr_atm_corr.case_catalog`` so there is a single source of truth.
This module only adds the extras that are specific to the cloud-radiative-effect
sweeps and have no home in that catalog: the solar-zenith-angle grid, the cloud
water-path sweep, and the cross-case ``manual_alb`` spectra used to compare CRE
under different surface albedos.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Catalog case ids to run CRE simulations / plots for.
# These are ids in ssfr_atm_corr.case_catalog (good + bad lists). They are the
# cloudy atmospheric-correction cases the legacy ssfr_cre script ran.
# ---------------------------------------------------------------------------
CRE_CASE_IDS = [
    'case_004',       # 2024-06-03 cloudy_atm_corr_2 (present in combined product)
    'case_019',       # 2024-06-13 cloudy_atm_corr_1 (100 m, cloudy)
    'bad_case_003',   # 2024-06-03 cloudy_atm_corr_1 (300 m, camera icing; not in
                      # the combined product -> cre_sim falls back to per-leg pickles)
]

DEFAULT_CRE_CASE_ID = CRE_CASE_IDS[0]


# ---------------------------------------------------------------------------
# Solar-zenith-angle sweep (degrees).
# The legacy code ran this in small chunks on the cluster; the full grid is
# kept here. ``cre_sza_array`` optionally appends the case-mean SZA.
# ---------------------------------------------------------------------------
CRE_SZA_GRID = np.array(
    [50, 52.5, 55, 57.5, 60, 62.5, 65, 67.5, 70, 71.5, 72.5, 73, 73.5, 75],
    dtype=np.float32,
)


def cre_sza_array(sza_avg=None):
    """Return the SZA sweep, optionally including the case-mean SZA."""
    if sza_avg is None:
        return CRE_SZA_GRID.copy()
    return np.unique(
        np.concatenate((CRE_SZA_GRID, np.array([np.round(sza_avg, 2)], dtype=np.float32)))
    )


# ---------------------------------------------------------------------------
# Cloud water-path sweep (g/m^2) used to build the COT grid. The per-case cloud
# water path from the catalog is appended at run time inside cre_sim.
# ---------------------------------------------------------------------------
CRE_CWP_LIST_MAC = [0, 5, 10, 30, 50, 100, 200]
CRE_CWP_LIST_LINUX = [
    0, 1, 2, 3, 5, 7.5, 10, 15, 20, 35, 50, 75, 100, 150, 200, 300, 400, 500, 600,
]


# ---------------------------------------------------------------------------
# Cross-case manual albedo spectra (filenames under data/sfc_alb_cre/).
# Used by cre_plot to compare CRE under a range of measured surface albedos.
#
# Selected from data/sfc_alb_cre/ext_alb_broadband.csv to span the broadband
# albedo range (~0.29-0.80) with minimal redundancy: near-identical broadband
# values (gap < ~0.02) were dropped, keeping one representative each. The inline
# value is the solar-flux-weighted broadband albedo.
# ---------------------------------------------------------------------------
MANUAL_ALB_SWEEP = [
    'sfc_alb_20240801_13.843_14.351_0.11km_cre_alb.dat',             # 0.294
    'sfc_alb_20240725_15.881_15.903_0.33km_cre_alb.dat',             # 0.543
    'sfc_alb_20240808_13.212_13.345_0.12km_cre_alb.dat',             # 0.564
    'sfc_alb_20240808_15.314_15.497_0.12km_cre_alb_scale_1.012X.dat',  # 0.608
    'sfc_alb_20240809_16.029_16.224_0.11km_cre_alb.dat',             # 0.638
    'sfc_alb_20240611_14.968_15.347_0.41km_cre_alb.dat',             # 0.672
    'sfc_alb_20240613_14.109_14.140_0.11km_cre_alb.dat',             # 0.676  (case_019 own albedo)
    'sfc_alb_20240528_15.610_17.404_0.18km_cre_alb.dat',             # 0.699
    'sfc_alb_20240603_14.711_14.868_0.34km_cre_alb.dat',             # 0.748  (case_004 own albedo)
    'sfc_alb_20240607_15.336_15.761_0.12km_cre_alb.dat',             # 0.752
    'sfc_alb_20240603_14.735_14.752_0.34km_cre_alb.dat',             # 0.758  (case_004 peak 1-min albedo)
    'sfc_alb_20240606_16.250_16.950_0.11km_cre_alb.dat',             # 0.776
    'sfc_alb_20240606_16.250_16.950_1.18km_cre_alb.dat',             # 0.797
]
