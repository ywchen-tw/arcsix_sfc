"""Date-aware post-fit albedo cleanup helpers."""

import numpy as np
import sys
from pathlib import Path
from scipy.ndimage import uniform_filter1d

_UTIL_ROOT = str(Path(__file__).resolve().parents[1] / 'util')
if _UTIL_ROOT not in sys.path:
    sys.path.insert(0, _UTIL_ROOT)

from alb_fitting import snowice_alb_fitting

try:
    from .settings import h2o_6_end, h2o_6_start, h2o_7_end, h2o_7_start
except ImportError:
    from settings import h2o_6_end, h2o_6_start, h2o_7_end, h2o_7_start


DATE_H2O_6_END = {
    '20240807': {'mask': 1550, 'fit': 1570},
}
POSTFIT_0603_DATES = {'20240603'}


def postfit_h2o_6_mask_end(date_s, default_end):
    """Return date-aware H2O-6 mask end for albedo masking."""
    override = DATE_H2O_6_END.get(str(date_s), {}).get('mask')
    return default_end if override is None else override


def postfit_h2o_6_fit_end(date_s, default_end):
    """Return date-aware H2O-6 fit end for snow/ice albedo fitting."""
    override = DATE_H2O_6_END.get(str(date_s), {}).get('fit')
    return default_end if override is None else override


def use_full_postfit_window(date_s):
    """Return True for dates whose post-fit cleanup may touch the full legacy span."""
    date_key = str(date_s)
    return date_key in POSTFIT_0603_DATES or date_key in DATE_H2O_6_END


def _linear_between(x0, y0, x1, y1, x):
    """Return linear interpolation/extrapolation between two finite anchors."""
    if x1 == x0:
        return np.full_like(np.asarray(x, dtype=float), y0, dtype=float)
    slope = (y1 - y0) / (x1 - x0)
    return y0 + slope * (np.asarray(x, dtype=float) - x0)


def _postfit_h2o6_h2o7_window(native_wvl, fitted_row):
    """Mirror legacy post-fit cleanup between H2O-6 and H2O-7."""
    wvl = np.asarray(native_wvl, dtype=float)
    values = np.asarray(fitted_row, dtype=float).copy()
    if values.size != wvl.size or np.all(~np.isfinite(values)):
        return values

    wvl1450_ind = int(np.argmin(np.abs(wvl - 1450)))
    wvl1520_ind = int(np.argmin(np.abs(wvl - 1520)))
    wvl1650_ind = int(np.argmin(np.abs(wvl - 1650)))
    wvl1710_ind = int(np.argmin(np.abs(wvl - 1710)))
    wvl1800_ind = int(np.argmin(np.abs(wvl - 1800)))

    if wvl1450_ind >= wvl1520_ind or wvl1800_ind >= values.size:
        return np.clip(values, 0.0, 1.0)

    start_window = values[wvl1450_ind:wvl1520_ind]
    start_valid = np.isfinite(start_window)
    if np.count_nonzero(start_valid) == 0:
        return np.clip(values, 0.0, 1.0)
    start_local = np.flatnonzero(start_valid)[np.argmin(start_window[start_valid])]
    start_wvl = wvl[wvl1450_ind:wvl1520_ind][start_local]
    start_ind = int(np.argmin(np.abs(wvl - start_wvl))) + 1

    end_window = values[wvl1800_ind:]
    end_valid = np.isfinite(end_window)
    if np.count_nonzero(end_valid) == 0:
        return np.clip(values, 0.0, 1.0)
    end_local = np.flatnonzero(end_valid)[np.argmax(end_window[end_valid])]
    end_wvl = wvl[wvl1800_ind:][end_local]
    end_ind = int(np.argmin(np.abs(wvl - end_wvl))) - 1

    if start_ind < 0 or end_ind >= values.size or start_ind >= end_ind:
        return np.clip(values, 0.0, 1.0)
    if not np.isfinite(values[start_ind]) or not np.isfinite(values[end_ind]):
        return np.clip(values, 0.0, 1.0)

    def fit_1d(x):
        return _linear_between(start_wvl, values[start_ind], end_wvl, values[end_ind], x)

    interp_slice = slice(wvl1650_ind, wvl1710_ind + 1)
    interp_wvl = wvl[interp_slice]
    if interp_wvl.size:
        interp_alb = np.clip(fit_1d(interp_wvl), 0.0, 1.0)
        compare_alb = values[interp_slice]
        valid = np.isfinite(compare_alb) & np.isfinite(interp_alb) & (np.abs(interp_alb) > 1e-12)
        if np.any(np.abs((compare_alb[valid] - interp_alb[valid]) / interp_alb[valid]) > 0.05):
            values[interp_slice] = interp_alb

    min_val = values[start_ind]
    segment = values[start_ind:end_ind + 1].copy()
    segment[np.isfinite(segment) & (segment < min_val)] = min_val
    values[start_ind:end_ind + 1] = segment

    compare_alb = values[start_ind:end_ind + 1]
    baseline = fit_1d(wvl[start_ind:end_ind + 1])
    valid = np.isfinite(compare_alb) & np.isfinite(baseline) & (np.abs(baseline) > 1e-12)
    if np.any(valid):
        diff_alb = np.full(compare_alb.shape, np.nan, dtype=float)
        diff_alb[valid] = np.abs((compare_alb[valid] - baseline[valid]) / baseline[valid])
        diff_std = np.nanstd(diff_alb)
        if diff_std > 0.1:
            values[start_ind:end_ind + 1] = (compare_alb + baseline * 7.0) / 8.0
        elif diff_std > 0.05:
            values[start_ind:end_ind + 1] = (compare_alb + baseline * 3.0) / 4.0
        elif diff_std > 0.02:
            values[start_ind:end_ind + 1] = (compare_alb + baseline) / 2.0

    smoothed = uniform_filter1d(values[start_ind:].copy(), size=5, mode='reflect')
    values[start_ind:] = np.clip(smoothed, 0.0, 1.0)
    return np.clip(values, 0.0, 1.0)


def _linear_fit_window(wvl, values, start_wvl, end_wvl):
    """Return a robust first-order fit for finite values in one wavelength window."""
    window = (wvl >= start_wvl) & (wvl <= end_wvl) & np.isfinite(values)
    x = wvl[window]
    y = values[window]
    if x.size < 2:
        return None

    coeffs = np.polyfit(x, y, 1)
    if x.size >= 5:
        fitted = np.poly1d(coeffs)(x)
        resid = y - fitted
        mad = np.nanmedian(np.abs(resid - np.nanmedian(resid)))
        sigma = 1.4826 * mad if mad > 0 else np.nanstd(resid)
        if np.isfinite(sigma) and sigma > 0:
            keep = np.abs(resid) <= 3 * sigma
            if np.count_nonzero(keep) >= 2:
                coeffs = np.polyfit(x[keep], y[keep], 1)
    return np.poly1d(coeffs)


def _refit_h2o6_band_with_snowice(native_wvl, fitted_row, alt, clear_sky):
    """Refit only the default H2O-6 band using the snow/ice fitting routine."""
    wvl = np.asarray(native_wvl, dtype=float)
    values = np.asarray(fitted_row, dtype=float).copy()
    if values.size != wvl.size or np.all(~np.isfinite(values)):
        return values

    band = (wvl >= h2o_6_start) & (wvl <= h2o_6_end)
    if np.count_nonzero(band) == 0:
        return np.clip(values, 0.0, 1.0)

    try:
        refitted = snowice_alb_fitting(
            wvl,
            values,
            alt=alt,
            clear_sky=clear_sky,
            h2o_6_end=h2o_6_end,
        )
    except Exception:
        return np.clip(values, 0.0, 1.0)
    refitted = np.asarray(refitted, dtype=float)
    if refitted.shape != values.shape or np.all(~np.isfinite(refitted[band])):
        return np.clip(values, 0.0, 1.0)
    replace = np.isfinite(refitted) & band
    values[replace] = refitted[replace]
    return np.clip(values, 0.0, 1.0)


def _postfit_0603_before_1650_correction(native_wvl, fitted_row, alt, clear_sky):
    """0603-only correction using the clean 1650-1740 nm continuum as anchor."""
    wvl = np.asarray(native_wvl, dtype=float)
    values = np.asarray(fitted_row, dtype=float).copy()
    if values.size != wvl.size or np.all(~np.isfinite(values)):
        return values

    continuum_fit = _linear_fit_window(wvl, values, 1650, 1740)
    if continuum_fit is None:
        return np.clip(values, 0.0, 1.0)

    continuum = (wvl > h2o_6_end) & (wvl <= 1740)
    if np.any(continuum):
        values[continuum] = continuum_fit(wvl[continuum])

    values = _refit_h2o6_band_with_snowice(
        wvl,
        values,
        alt=alt,
        clear_sky=clear_sky,
    )
    return np.clip(values, 0.0, 1.0)


def _postfit_h2o6_to_h2o7_from_1495(
    native_wvl,
    fitted_row,
    h2o6_end,
    h2o7_start,
    h2o7_end=h2o_7_end,
    min_wvl=1495.0,
    right_anchor_start=1700.0,
    right_anchor_end=1740.0,
    left_taper_nm=30.0,
    right_taper_nm=30.0,
    h2o7_taper_nm=30.0,
    h2o7_smooth_size=9,
    h2o7_anchor_wvl=1835.0,
    h2o7_anchor_start=1800.0,
):
    """Constrain the 1495-H2O7 continuum and smooth the H2O-7 region."""
    wvl = np.asarray(native_wvl, dtype=float)
    original = np.asarray(fitted_row, dtype=float)
    values = original.copy()
    if values.size != wvl.size or np.all(~np.isfinite(values)):
        return values
    if not np.isfinite(h2o6_end) or not np.isfinite(h2o7_start) or h2o6_end >= h2o7_start:
        return np.clip(values, 0.0, 1.0)

    min_ind = int(np.argmin(np.abs(wvl - min_wvl)))
    if not np.isfinite(original[min_ind]):
        local_min = (wvl >= 1400.0) & (wvl <= 1500.0) & np.isfinite(original)
        if np.count_nonzero(local_min) == 0:
            return np.clip(values, 0.0, 1.0)
        local_indices = np.flatnonzero(local_min)
        min_ind = local_indices[int(np.argmin(original[local_min]))]

    min_anchor_wvl = float(wvl[min_ind])
    min_anchor_val = float(original[min_ind])
    right_anchor_stop = min(float(right_anchor_end), float(h2o7_start))
    right_anchor = (
        (wvl >= right_anchor_start)
        & (wvl <= right_anchor_stop)
        & np.isfinite(original)
    )

    if np.count_nonzero(right_anchor) >= 2:
        right_fit = _linear_fit_window(wvl, original, right_anchor_start, right_anchor_stop)
        if right_fit is None:
            return np.clip(values, 0.0, 1.0)
        right_wvl = right_anchor_stop
        right_val = float(right_fit(right_wvl))
    elif np.count_nonzero(right_anchor) == 1:
        right_ind = np.flatnonzero(right_anchor)[0]
        right_wvl = float(wvl[right_ind])
        right_val = float(original[right_ind])
    else:
        finite_right = (wvl > min_anchor_wvl) & (wvl < h2o7_start) & np.isfinite(original)
        if np.count_nonzero(finite_right) == 0:
            return np.clip(values, 0.0, 1.0)
        right_ind = np.flatnonzero(finite_right)[-1]
        right_wvl = float(wvl[right_ind])
        right_val = float(original[right_ind])

    if not np.isfinite(right_val) or right_wvl <= min_anchor_wvl:
        return np.clip(values, 0.0, 1.0)
    right_val = max(right_val, min_anchor_val)
    h2o7_anchor_val = float(
        _linear_between(min_anchor_wvl, min_anchor_val, right_wvl, right_val, h2o7_anchor_wvl)
    )
    h2o7_anchor_val = max(h2o7_anchor_val, min_anchor_val)

    replace = (wvl > h2o6_end) & (wvl < h2o7_start) & np.isfinite(original)
    if np.count_nonzero(replace) == 0:
        return np.clip(values, 0.0, 1.0)

    baseline = _linear_between(min_anchor_wvl, min_anchor_val, right_wvl, right_val, wvl[replace])
    baseline = np.maximum(baseline, min_anchor_val)

    alpha = np.ones(np.count_nonzero(replace), dtype=float)
    replace_wvl = wvl[replace]
    if left_taper_nm > 0:
        alpha = np.minimum(alpha, np.clip((replace_wvl - h2o6_end) / left_taper_nm, 0.0, 1.0))
    if right_taper_nm > 0:
        alpha = np.minimum(alpha, np.clip((h2o7_start - replace_wvl) / right_taper_nm, 0.0, 1.0))

    values[replace] = (1.0 - alpha) * original[replace] + alpha * baseline
    values = _smooth_h2o7_region(
        wvl,
        values,
        original,
        h2o7_start=h2o7_start,
        h2o7_end=h2o7_end,
        taper_nm=h2o7_taper_nm,
        smooth_size=h2o7_smooth_size,
        anchor_wvl=h2o7_anchor_wvl,
        anchor_val=h2o7_anchor_val,
        anchor_start=h2o7_anchor_start,
    )
    return np.clip(values, 0.0, 1.0)


def _smooth_h2o7_region(
    wvl,
    values,
    original,
    h2o7_start,
    h2o7_end,
    taper_nm=30.0,
    smooth_size=9,
    anchor_wvl=None,
    anchor_val=None,
    anchor_start=1800.0,
):
    """Smooth fitted albedo after H2O-7 starts while blending in at the boundary."""
    values = np.asarray(values, dtype=float).copy()
    original = np.asarray(original, dtype=float)
    h2o7 = (
        (wvl >= h2o7_start)
        & (wvl <= h2o7_end)
        & np.isfinite(original)
    )
    if np.count_nonzero(h2o7) < 5:
        return values

    x = wvl[h2o7]
    y = values[h2o7].copy()
    finite = np.isfinite(y)
    if np.count_nonzero(finite) < 5:
        return values
    if not np.all(finite):
        y[~finite] = np.interp(x[~finite], x[finite], y[finite])

    smooth_size = max(3, int(smooth_size))
    if smooth_size % 2 == 0:
        smooth_size += 1

    continuum = uniform_filter1d(y, size=smooth_size, mode='reflect')
    resid = y - continuum
    mad = np.nanmedian(np.abs(resid - np.nanmedian(resid)))
    sigma = 1.4826 * mad if mad > 0 else np.nanstd(resid)
    if np.isfinite(sigma) and sigma > 0:
        spike = np.abs(resid) > 3 * sigma
        y[spike] = continuum[spike]

    smoothed = uniform_filter1d(y, size=smooth_size, mode='reflect')
    if anchor_wvl is not None and anchor_val is not None and np.isfinite(anchor_val):
        anchor_ind = int(np.argmin(np.abs(x - anchor_wvl)))
        if np.isfinite(smoothed[anchor_ind]):
            offset = float(anchor_val) - float(smoothed[anchor_ind])
            weights = np.zeros_like(smoothed)
            if anchor_wvl > anchor_start:
                left = (x >= anchor_start) & (x <= anchor_wvl)
                weights[left] = (x[left] - anchor_start) / (anchor_wvl - anchor_start)
            right_denom = h2o7_end - anchor_wvl
            if right_denom > 0:
                right = x > anchor_wvl
                weights[right] = (h2o7_end - x[right]) / right_denom
            weights = np.clip(weights, 0.0, 1.0)
            smoothed = smoothed + offset * weights

    alpha = np.ones_like(smoothed)
    if taper_nm > 0:
        alpha = np.clip((x - h2o7_start) / taper_nm, 0.0, 1.0)
        tail_taper = np.clip((h2o7_end - x) / taper_nm, 0.0, 1.0)
        alpha = np.minimum(alpha, tail_taper)

    h2o7_values = (1.0 - alpha) * original[h2o7] + alpha * smoothed
    values[h2o7] = h2o7_values
    return values


def apply_postfit_correction(
    native_wvl,
    fitted_row,
    date_s=None,
    alt=np.nan,
    clear_sky=False,
    preserve_outside_window=False,
    window_start=None,
    window_end=None,
):
    """Apply the standard post-fit cleanup, plus date-specific cleanup when needed."""
    if preserve_outside_window:
        start = h2o_6_end if window_start is None else window_start
        end = h2o_7_start if window_end is None else window_end
        corrected = _postfit_h2o6_to_h2o7_from_1495(
            native_wvl,
            fitted_row,
            h2o6_end=start,
            h2o7_start=end,
        )
    else:
        corrected = _postfit_h2o6_h2o7_window(native_wvl, fitted_row)
    if str(date_s) in POSTFIT_0603_DATES:
        corrected = _postfit_0603_before_1650_correction(
            native_wvl,
            corrected,
            alt=alt,
            clear_sky=clear_sky,
        )
    return np.clip(corrected, 0.0, 1.0)
