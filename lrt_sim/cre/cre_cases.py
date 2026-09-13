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
    'case_014',       # 2024-06-07 cloudy_atm_corr (present in combined product)
    'case_019',       # 2024-06-13 cloudy_atm_corr_1 (100 m, cloudy)
    'bad_case_003',   # 2024-06-03 cloudy_atm_corr_1 (300 m, camera icing; not in
                      # the combined product -> cre_sim falls back to per-leg pickles)
]

DEFAULT_CRE_CASE_ID = CRE_CASE_IDS[0]


# ---------------------------------------------------------------------------
# Solar-zenith-angle sweep (degrees). SINGLE SOURCE OF TRUTH: both the
# simulation (cre_sim) and the post-processing (cre_plot) build their SZA axis
# from ``cre_sza_array`` below, so editing this grid changes what is simulated
# and what is plotted together. Every angle here is run; nothing beyond it is
# (the old 77.5-87 tail was simulated but never read by cre_plot).
# ---------------------------------------------------------------------------
CRE_SZA_GRID = np.array(
    [50, 52.5, 55, 57.5, 60, 62.5, 63.75, 65, 66.25, 67.5, 70, 71.5, 72.5, 73, 73.5, 75],
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
# SZA chunks: split one (case, albedo) cluster job into several smaller runs so
# each fits comfortably inside a wall-clock limit. Their union is exactly
# CRE_SZA_GRID plus the case-mean SZA -- i.e. what cre_plot reads -- and
# ``_validate_sza_chunks`` below enforces that on import, so a chunk edit can
# never silently drop an angle from the plots.
#
# The case-mean SZA is computed at run time from the flight data, so it cannot
# be written as a literal; ``include_avg`` marks the one chunk that also runs it.
# ---------------------------------------------------------------------------
CRE_SZA_CHUNKS = [
    {'sza': [50, 52.5, 55, 57.5],    'include_avg': False},
    {'sza': [60, 62.5, 63.75, 65],   'include_avg': False},
    {'sza': [66.25, 67.5, 70, 71.5], 'include_avg': False},
    {'sza': [72.5, 73, 73.5, 75],    'include_avg': True},
]


def _validate_sza_chunks():
    """Fail loudly on import if the chunks drift from CRE_SZA_GRID."""
    covered = sorted(float(v) for chunk in CRE_SZA_CHUNKS for v in chunk['sza'])
    grid = sorted(float(v) for v in CRE_SZA_GRID)
    if covered != grid:
        missing = [v for v in grid if v not in covered]
        extra = [v for v in covered if v not in grid]
        raise ValueError(
            'CRE_SZA_CHUNKS must cover CRE_SZA_GRID exactly. '
            f'Missing from chunks: {missing}. Not in grid: {extra}.'
        )
    n_avg = sum(bool(chunk['include_avg']) for chunk in CRE_SZA_CHUNKS)
    if n_avg != 1:
        raise ValueError(
            f'Exactly one CRE_SZA_CHUNKS entry must set include_avg; found {n_avg}. '
            'Without it the case-mean SZA is never simulated and cre_plot loses an angle.'
        )


_validate_sza_chunks()


def sza_chunk(index):
    """Return ``(sza_list, include_avg)`` for one chunk index."""
    try:
        chunk = CRE_SZA_CHUNKS[index]
    except IndexError:
        raise IndexError(
            f'SZA chunk index {index} out of range; '
            f'CRE_SZA_CHUNKS has {len(CRE_SZA_CHUNKS)} entries (0-{len(CRE_SZA_CHUNKS) - 1}).'
        ) from None
    return list(chunk['sza']), bool(chunk['include_avg'])


# ---------------------------------------------------------------------------
# Cloud water-path sweep (g/m^2) used to build the COT grid. The per-case cloud
# water path from the catalog is appended at run time inside cre_sim.
# ---------------------------------------------------------------------------
CRE_CWP_LIST_MAC = [0, 5, 10, 30, 50, 100, 200]
CRE_CWP_LIST_LINUX = [
    0, 1, 2, 3, 5, 7.5, 10, 15, 20, 35, 50, 75, 100, 150, 200, 300, 350, 400, 450, 500, 600, 800
]


# ---------------------------------------------------------------------------
# Cross-case manual albedo spectra (filenames under data/sfc_alb_cre/).
# Used by cre_plot to compare CRE under a range of measured surface albedos.
#
# Selected from data/sfc_alb_cre/ext_alb_broadband.csv to span the broadband
# albedo range (~0.30-0.78) with minimal redundancy: near-identical broadband
# values (gap < ~0.02) were dropped, keeping one representative each. The inline
# value is the TOA-solar-flux-weighted broadband albedo, sorted ascending.
#
# Values refreshed 2026-09-12 from ext_alb_broadband.csv after the combined
# product was rebuilt; most moved by <0.02 but 20240611_..._0.12km fell 0.704 ->
# 0.666. The solar weight is the 250-4050 nm slit file (adopted 2026-07-17; the
# old 2500 nm file gave values ~0.013-0.019 higher).
#
# NOTE: several entries sit under the ~0.02 spacing rule above. The 2026-09-12
# refresh pushed two pairs back under it -- 0.610/0.611 (20240809_16.029 vs
# 20240808_scale_1.012X) and 0.661/0.666 (20240613_14.109 vs 20240611_..._0.12km)
# -- and the case_004 family is deliberately dense at 0.732/0.735/0.742 so the
# observation is bracketed by its own full-window and 3-min means. The 0.752
# entry likewise splits the 0.742-0.762 gap at its midpoint (0.010 either side).
# All are kept for now; thin them if the tight spacing kinks the critical-LWP
# contour.
# ---------------------------------------------------------------------------
MANUAL_ALB_SWEEP = [
    'sfc_alb_20240801_13.843_14.351_0.11km_cre_alb.dat',             # 0.297
    'sfc_alb_20240725_15.881_15.903_0.33km_cre_alb.dat',             # 0.531
    'sfc_alb_20240808_13.212_13.345_0.12km_cre_alb.dat',             # 0.568
    'sfc_alb_20240809_16.029_16.224_0.11km_cre_alb.dat',             # 0.610
    'sfc_alb_20240808_15.314_15.497_0.12km_cre_alb_scale_1.012X.dat',  # 0.611
    'sfc_alb_20240611_14.968_15.347_0.41km_cre_alb.dat',             # 0.641
    'sfc_alb_20240613_14.109_14.140_0.11km_cre_alb.dat',             # 0.661  (case_019 own albedo)
    'sfc_alb_20240611_14.968_15.347_0.12km_cre_alb.dat',             # 0.666
    'sfc_alb_20240528_15.610_17.404_0.18km_cre_alb.dat',             # 0.686
    # 0.694 (sfc_alb_20240613_15.834_..._scale_0.987X) dropped: thinned the tight
    # cluster near 0.7 that kinked the critical-LWP contour.
    # The case_004 family below spans 0.732-0.742. The peak 1-min albedo (0.742)
    # stays commented out: it collides with the 2-min broadband bin and corrupts
    # that albedo's critical-LWP column.
    # 'sfc_alb_20240603_14.735_14.752_0.34km_cre_alb.dat',           # 0.742  (case_004 peak 1-min albedo)
    'sfc_alb_20240603_14.711_14.868_0.34km_cre_alb.dat',             # 0.732  (case_004 full-window mean)
    'sfc_alb_20240603_14.711_14.761_0.34km_cre_alb.dat',             # 0.735  (case_004 peak 3-min albedo)
    'sfc_alb_20240603_14.716_14.749_0.34km_cre_alb.dat',             # 0.742  (case_004 peak 2-min albedo, observation)
    # Only entry above ~1.2 km: a 5.8 km mean over the whole 20240605 window, so
    # its surface albedo comes from a longer correction path than the rest. The
    # same date's low legs give 0.762 (0.44 km) and 0.766 (0.11 km).
    'sfc_alb_20240605_12.422_13.812_5.80km_cre_alb.dat',             # 0.752
    'sfc_alb_20240606_16.250_16.950_0.11km_cre_alb.dat',             # 0.762
    'sfc_alb_20240606_16.250_16.950_1.18km_cre_alb.dat',             # 0.776
]
