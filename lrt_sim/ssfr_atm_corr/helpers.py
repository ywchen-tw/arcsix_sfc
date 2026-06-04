"""Small helpers for SSFR atmospheric correction."""

from enum import IntFlag, auto

import numpy as np
import pandas as pd

try:
    from .settings import (
        final_end,
        final_start,
        h2o_1_end,
        h2o_1_start,
        h2o_2_end,
        h2o_2_start,
        h2o_3_end,
        h2o_3_start,
        h2o_4_end,
        h2o_4_start,
        h2o_5_end,
        h2o_5_start,
        h2o_6_end,
        h2o_6_start,
        h2o_7_end,
        h2o_7_start,
        h2o_8_end,
        h2o_8_start,
        o2a_1_end,
        o2a_1_start,
    )
except ImportError:
    from settings import (
        final_end,
        final_start,
        h2o_1_end,
        h2o_1_start,
        h2o_2_end,
        h2o_2_start,
        h2o_3_end,
        h2o_3_start,
        h2o_4_end,
        h2o_4_start,
        h2o_5_end,
        h2o_5_start,
        h2o_6_end,
        h2o_6_start,
        h2o_7_end,
        h2o_7_start,
        h2o_8_end,
        h2o_8_start,
        o2a_1_end,
        o2a_1_start,
    )


def fit_1d_poly(x, y, order=1, dx=None, x0=None):
    """Fit a linear endpoint trend and return it as a poly1d object."""
    mask = ~np.isnan(x) & ~np.isnan(y)
    coeffs = np.polyfit(x[mask], y[mask], order)

    if x0 is None:
        x0 = np.nanmean(x[mask][:2])
    y0 = np.nanmean(y[mask][:2])
    x1 = np.nanmean(x[mask][-2:])
    y1 = np.nanmean(y[mask][-2:])
    if dx is None:
        dx = x1 - x0
    slope = (y1 - y0) / dx
    intercept = y0 - slope * x0
    coeffs = [slope, intercept]

    return np.poly1d(coeffs)


class ssfr_flags(IntFlag):
    pitcth_roll_exceed_threshold = auto()
    camera_icing = auto()
    camera_icing_pre = auto()
    zen_toa_over_threshold = auto()
    alp_ang_pit_rol_issue = auto()


def write_2col_file(filename, wvl, val, header):
    """Write wavelength/value arrays as a two-column text file."""
    with open(filename, 'w') as f:
        f.write(header)
        for i in range(len(val)):
            f.write(f'{wvl[i]:11.3f} {val[i]:12.3e}\n')


def find_h2o_6_end(wvl, alb, default_end=h2o_6_end):
    """Extend the H2O-6 mask until albedo recovers to its pre-band value."""
    wvl = np.asarray(wvl, dtype=float)
    alb = np.asarray(alb, dtype=float)
    finite = np.isfinite(wvl) & np.isfinite(alb)

    before_indices = np.flatnonzero(finite & (wvl < h2o_6_start))
    after_indices = np.flatnonzero(finite & (wvl > default_end))
    if before_indices.size == 0 or after_indices.size == 0:
        return default_end

    reference_albedo = alb[before_indices[-1]]
    first_after_index = after_indices[0]
    if alb[first_after_index] <= reference_albedo:
        return default_end

    recovered_indices = after_indices[alb[after_indices] <= reference_albedo]
    if recovered_indices.size == 0:
        return default_end
    return float(wvl[recovered_indices[0]])


def gas_abs_masking(
    wvl,
    alb,
    alt,
    altitude_dependent=False,
    h2o_6_end_override=None,
):
    """Mask all gas bands by default, with optional reduced low-altitude masking."""
    effective_mask_ = np.ones_like(alb)
    alb_mask = alb.copy()
    selected_h2o_6_end = h2o_6_end if h2o_6_end_override is None else h2o_6_end_override
    full_mask = (
        ((wvl >= o2a_1_start) & (wvl <= o2a_1_end))
        | ((wvl >= h2o_1_start) & (wvl <= h2o_1_end))
        | ((wvl >= h2o_2_start) & (wvl <= h2o_2_end))
        | ((wvl >= h2o_3_start) & (wvl <= h2o_3_end))
        | ((wvl >= h2o_4_start) & (wvl <= h2o_4_end))
        | ((wvl >= h2o_5_start) & (wvl <= h2o_5_end))
        | ((wvl >= h2o_6_start) & (wvl <= selected_h2o_6_end))
        | ((wvl >= h2o_7_start) & (wvl <= h2o_7_end))
        | ((wvl >= h2o_8_start) & (wvl <= h2o_8_end))
        | ((wvl >= final_start) & (wvl <= final_end))
    )

    # Future option: retain O2 masking but omit short-path H2O bands below 0.5 km.
    reduced_low_altitude_mask = (
        ((wvl >= o2a_1_start) & (wvl <= o2a_1_end))
        | ((wvl >= h2o_3_start) & (wvl <= h2o_3_end))
        | ((wvl >= h2o_4_start) & (wvl <= h2o_4_end))
        | ((wvl >= h2o_5_start) & (wvl <= h2o_5_end))
        | ((wvl >= h2o_6_start) & (wvl <= selected_h2o_6_end))
        | ((wvl >= h2o_7_start) & (wvl <= h2o_7_end))
        | ((wvl >= final_start) & (wvl <= final_end))
    )
    mask = reduced_low_altitude_mask if altitude_dependent and alt <= 0.5 else full_mask

    alb_mask[mask] = np.nan
    effective_mask_[mask] = np.nan

    if np.sum(~np.isnan(effective_mask_)) != np.isfinite(alb_mask).sum():
        fit_wvl_mask = np.logical_and(~np.isnan(effective_mask_), np.isnan(alb_mask))

        s = pd.Series(alb_mask[effective_mask_ == 1])
        s_mask = np.isnan(alb_mask[effective_mask_ == 1])
        s_ffill = s.ffill(limit=2).bfill(limit=2)
        while np.any(np.isnan(s_ffill)):
            s_ffill = s_ffill.ffill(limit=2).bfill(limit=2)

        alb_mask[fit_wvl_mask] = np.array(s_ffill)[s_mask]

    return alb_mask
