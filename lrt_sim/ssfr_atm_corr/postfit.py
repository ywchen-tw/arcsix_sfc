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
    from .settings import h2o_6_end, h2o_6_start
except ImportError:
    from settings import h2o_6_end, h2o_6_start


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


def apply_postfit_correction(native_wvl, fitted_row, date_s=None, alt=np.nan, clear_sky=False):
    """Apply the standard post-fit cleanup, plus date-specific cleanup when needed."""
    corrected = _postfit_h2o6_h2o7_window(native_wvl, fitted_row)
    if str(date_s) in POSTFIT_0603_DATES:
        corrected = _postfit_0603_before_1650_correction(
            native_wvl,
            corrected,
            alt=alt,
            clear_sky=clear_sky,
        )
    return np.clip(corrected, 0.0, 1.0)
