"""Post-process SSFR atmospheric-correction products.

This module consumes products written by ``ssfr_atm_corr.workflow``:

* ``*_final.csv`` on the native SSFR wavelength grid.
* ``*_final_extension.csv`` on the extended 300-4000 nm grid.
* ``sfc_alb_*_final.dat`` on the native albedo grid.
* ``sfc_alb_*_final_extension.dat`` on the extended albedo grid.
"""

import datetime
import gc
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_LRT_SIM_ROOT = str(Path(__file__).resolve().parents[1])
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
_UTIL_ROOT = str(Path(__file__).resolve().parents[1] / 'util')
_SSFR_ROOT = str(Path(__file__).resolve().parent)
for _path in (_SSFR_ROOT, _UTIL_ROOT, _REPO_ROOT, _LRT_SIM_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

from alb_fitting import alb_extention, snowice_alb_fitting

DATE_H2O_6_END = {
    '20240603': {'mask': 1650, 'fit': 1650},
    '20240807': {'mask': 1550, 'fit': 1570},
}
POSTFIT_H2O6_H2O7_DATES = {'20240603', '20240807', '20240808', '20240809'}
REFIT_H2O6_AFTER_POSTFIT_DATES = {'20240603'}
POSTFIT_DIAGNOSTIC_STRIDE = 100

try:
    from .helpers import gas_abs_masking
    from .settings import _fdir_data_, _fdir_general_, gas_bands, h2o_6_end, h2o_6_start
except ImportError:
    from helpers import gas_abs_masking
    from settings import _fdir_data_, _fdir_general_, gas_bands, h2o_6_end, h2o_6_start


@dataclass(frozen=True)
class FlightConfig:
    mission: str
    platform: str
    data_root: Path
    root_mac: Path
    root_linux: Path

    def hsk(self, date_s): return f"{self.data_root}/{self.mission}-HSK_{self.platform}_{date_s}_v0.h5"
    def ssfr(self, date_s): return f"{self.data_root}/{self.mission}-SSFR_{self.platform}_{date_s}_R1.h5"
    def ssrr(self, date_s): return f"{self.data_root}/{self.mission}-SSRR_{self.platform}_{date_s}_R0.h5"
    def hsr1(self, date_s): return f"{self.data_root}/{self.mission}-HSR1_{self.platform}_{date_s}_R0.h5"
    def logic(self, date_s): return f"{self.data_root}/{self.mission}-LOGIC_{self.platform}_{date_s}_RA.h5"
    def sat_coll(self, date_s): return f"{self.data_root}/{self.mission}-SAT-CLD_{self.platform}_{date_s}_v0.h5"
    def marli(self, date_s):
        root = self.root_mac if sys.platform == "darwin" else self.root_linux
        return f"{root}/marli/ARCSIX-MARLi_P3B_{date_s}_R0.cdf"
    def kt19(self, date_s):
        root = self.root_mac if sys.platform == "darwin" else self.root_linux
        return f"{root}/kt19/ARCSIX-MetNav-KT19-10Hz_P3B_{date_s}_R0.ict"
    def sat_nc(self, date_s, raw):
        root = self.root_mac if sys.platform == "darwin" else self.root_linux
        return f"{root}/sat-data/{date_s}/{raw}"


def make_default_config():
    """Return the default ARCSIX P3B data-location config."""
    return FlightConfig(
        mission='ARCSIX',
        platform='P3B',
        data_root=_fdir_data_,
        root_mac=_fdir_general_,
        root_linux='/pl/active/vikas-arcsix/yuch8913/arcsix/data',
    )


def read_two_column_file(filename, value_name):
    """Read a two-column wavelength/value text file."""
    df = pd.read_csv(filename, sep=r'\s+', comment='#', names=['wvl', value_name])
    return df['wvl'].to_numpy(), df[value_name].to_numpy()


def maybe_read_two_column_file(filename, value_name):
    """Read a two-column file if it exists, otherwise return ``(None, None)``."""
    if not os.path.exists(filename):
        return None, None
    return read_two_column_file(filename, value_name)


def weighted_broadband_albedo(albedo, weight, wvl):
    """Return wavelength-integrated albedo weighted by spectral irradiance."""
    valid = np.isfinite(albedo) & np.isfinite(weight) & np.isfinite(wvl)
    if np.count_nonzero(valid) < 2:
        return np.nan
    denominator = np.trapz(weight[valid], x=wvl[valid])
    if denominator == 0:
        return np.nan
    return np.trapz(albedo[valid] * weight[valid], x=wvl[valid]) / denominator


def weighted_broadband_albedo_rows(albedo, weight, wvl):
    """Return per-row wavelength-integrated albedo weighted by spectral irradiance."""
    albedo = np.asarray(albedo, dtype=float)
    weight = np.asarray(weight, dtype=float)
    wvl = np.asarray(wvl, dtype=float)
    if albedo.ndim == 1:
        return weighted_broadband_albedo(albedo, weight, wvl)
    if weight.ndim == 1:
        weight = np.broadcast_to(weight, albedo.shape)

    broadband = np.full(albedo.shape[0], np.nan, dtype=float)
    finite_wvl = np.isfinite(wvl)
    valid = finite_wvl[np.newaxis, :] & np.isfinite(albedo) & np.isfinite(weight)

    fully_valid = np.all(valid[:, finite_wvl], axis=1)
    if np.any(fully_valid):
        denominator = np.trapz(weight[fully_valid][:, finite_wvl], x=wvl[finite_wvl], axis=1)
        numerator = np.trapz(
            albedo[fully_valid][:, finite_wvl] * weight[fully_valid][:, finite_wvl],
            x=wvl[finite_wvl],
            axis=1,
        )
        good = np.isfinite(denominator) & (denominator != 0)
        fully_valid_indices = np.flatnonzero(fully_valid)
        broadband[fully_valid_indices[good]] = numerator[good] / denominator[good]

    for irow in np.flatnonzero(~fully_valid):
        row_valid = valid[irow]
        if np.count_nonzero(row_valid) < 2:
            continue
        denominator = np.trapz(weight[irow, row_valid], x=wvl[row_valid])
        if denominator == 0 or not np.isfinite(denominator):
            continue
        broadband[irow] = (
            np.trapz(albedo[irow, row_valid] * weight[irow, row_valid], x=wvl[row_valid])
            / denominator
        )
    return broadband


def dataframe_weight(df, preferred_columns, expected_length):
    """Return the first available finite spectral weight column from a DataFrame."""
    for column in preferred_columns:
        if column in df and len(df[column]) == expected_length:
            weight = pd.to_numeric(df[column], errors='coerce').to_numpy()
            if np.any(np.isfinite(weight)):
                return weight
    return np.ones(expected_length, dtype=float)


def dataframe_column(df, column, expected_length):
    """Return one numeric DataFrame column or a NaN-filled spectrum."""
    if df is not None and column in df and len(df[column]) == expected_length:
        return pd.to_numeric(df[column], errors='coerce').to_numpy(dtype=float)
    return np.full(expected_length, np.nan, dtype=float)


def stack_or_object(values):
    """Stack equal-shaped arrays; otherwise keep an object array."""
    if not values:
        return np.array([])
    shapes = [np.shape(value) for value in values]
    if all(shape == shapes[0] for shape in shapes):
        return np.stack(values)
    return np.array(values, dtype=object)


def first_nonempty_array(records, key):
    """Return the first non-empty array stored under key in records."""
    for record in records:
        value = record[key]
        if np.size(value) > 0:
            return value
    return np.array([])


def concatenate_record_arrays(records, key):
    """Concatenate one-dimensional arrays stored in each record."""
    arrays = [np.asarray(record[key]) for record in records if np.size(record[key]) > 0]
    if not arrays:
        return np.array([])
    return np.concatenate(arrays)


def repeat_spectral_by_time(records, spectral_key, default_wvl):
    """Repeat one leg-mean spectrum for every sample time in each leg."""
    repeated = []
    for record in records:
        n_time = len(record['time'])
        spectrum = np.asarray(record[spectral_key], dtype=float)
        if spectrum.size == 0:
            spectrum = np.full(default_wvl.shape, np.nan, dtype=float)
        repeated.append(np.repeat(spectrum[np.newaxis, :], n_time, axis=0))
    if not repeated:
        return np.empty((0, len(default_wvl)))
    return np.vstack(repeated)


def repeat_scalar_by_time(records, scalar_key):
    """Repeat one leg-mean scalar for every sample time in each leg."""
    repeated = [
        np.full(len(record['time']), record[scalar_key], dtype=float)
        for record in records
    ]
    if not repeated:
        return np.array([])
    return np.concatenate(repeated)


def native_albedo_values(wvl, alb, native_wvl):
    """Map padded native albedo files onto the native CSV wavelength grid."""
    if wvl is None or alb is None:
        return np.full(native_wvl.shape, np.nan, dtype=float)
    if len(wvl) == len(native_wvl) + 2:
        return alb[1:-1]
    if len(wvl) == len(native_wvl) and np.allclose(wvl, native_wvl):
        return alb
    return np.interp(native_wvl, wvl, alb, left=np.nan, right=np.nan)


def fill_nan_ffill_bfill(arr, limit=2):
    """Fill NaNs in a one-dimensional array using repeated forward/backward fill."""
    values = np.asarray(arr, dtype=float)
    if not np.any(np.isnan(values)):
        return values
    if np.all(np.isnan(values)):
        return values
    series = pd.Series(values)
    series = series.ffill(limit=limit).bfill(limit=limit)
    while series.isna().any():
        series = series.ffill(limit=limit).bfill(limit=limit)
    return series.to_numpy()


def read_iteration_corr_factor(csv_path, native_wvl):
    """Read an iteration CSV correction factor on the native wavelength grid."""
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    if 'corr_factor' not in df:
        return None
    corr = pd.to_numeric(df['corr_factor'], errors='coerce').to_numpy(dtype=float)
    if corr.size == native_wvl.size:
        corr_native = corr
    elif 'wvl' in df:
        csv_wvl = pd.to_numeric(df['wvl'], errors='coerce').to_numpy(dtype=float)
        valid = np.isfinite(csv_wvl) & np.isfinite(corr)
        if np.count_nonzero(valid) < 2:
            return None
        corr_native = np.interp(native_wvl, csv_wvl[valid], corr[valid], left=np.nan, right=np.nan)
    else:
        return None

    if np.any(~np.isfinite(corr_native)):
        corr_native = fill_nan_ffill_bfill(corr_native)
    return np.where(np.isfinite(corr_native), corr_native, 1.0)


def apply_corr_factor_1s(albedo_1s, corr_factor):
    """Apply one spectral correction factor to every one-second albedo spectrum."""
    corrected = np.asarray(albedo_1s, dtype=float).copy()
    for irow in range(corrected.shape[0]):
        if np.any(np.isnan(corrected[irow])) and np.any(np.isfinite(corrected[irow])):
            corrected[irow] = fill_nan_ffill_bfill(corrected[irow])
    corrected = np.clip(corrected, 0.0, 1.0)
    corrected = corrected * corr_factor[np.newaxis, :]
    return np.clip(corrected, 0.0, 1.0)


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


def _refit_h2o6_band_from_postfit(native_wvl, fitted_row):
    """Refit the default H2O-6 band using neighboring post-fit albedo anchors."""
    wvl = np.asarray(native_wvl, dtype=float)
    values = np.asarray(fitted_row, dtype=float).copy()
    if values.size != wvl.size or np.all(~np.isfinite(values)):
        return values

    band = (wvl >= h2o_6_start) & (wvl <= h2o_6_end)
    if np.count_nonzero(band) == 0:
        return np.clip(values, 0.0, 1.0)

    finite = np.isfinite(values)
    left_candidates = np.flatnonzero(finite & (wvl < h2o_6_start))
    right_candidates = np.flatnonzero(finite & (wvl > h2o_6_end))
    if left_candidates.size == 0 or right_candidates.size == 0:
        return np.clip(values, 0.0, 1.0)

    left_ind = left_candidates[-1]
    right_ind = right_candidates[0]
    values[band] = _linear_between(
        wvl[left_ind],
        values[left_ind],
        wvl[right_ind],
        values[right_ind],
        wvl[band],
    )
    return np.clip(values, 0.0, 1.0)


def fit_albedo_1s(
    native_wvl,
    corrected_1s,
    alt_1s,
    clear_sky,
    date_s=None,
    postfit_diagnostics=None,
    time_1s=None,
    target_iter=None,
):
    """Apply snow/ice fitting to each one-second corrected albedo spectrum."""
    fitted = np.full_like(corrected_1s, np.nan, dtype=float)
    date_key = str(date_s)
    h2o_override = DATE_H2O_6_END.get(date_key, {})
    fit_kwargs = {'h2o_6_end': h2o_override['fit']} if 'fit' in h2o_override else {}
    apply_postfit_cleanup = date_key in POSTFIT_H2O6_H2O7_DATES
    refit_h2o6_after_postfit = date_key in REFIT_H2O6_AFTER_POSTFIT_DATES
    for irow in range(corrected_1s.shape[0]):
        row = corrected_1s[irow]
        if np.all(~np.isfinite(row)):
            continue
        alt = float(alt_1s[irow]) if np.isfinite(alt_1s[irow]) else np.nan
        try:
            fitted_row = snowice_alb_fitting(native_wvl, row, alt=alt, clear_sky=clear_sky, **fit_kwargs)
            if np.all(~np.isfinite(fitted_row)):
                raise ValueError('all non-finite fitted albedo')
            if apply_postfit_cleanup:
                before_postfit = fitted_row.copy()
                fitted_row = _postfit_h2o6_h2o7_window(native_wvl, fitted_row)
                if refit_h2o6_after_postfit:
                    fitted_row = _refit_h2o6_band_from_postfit(native_wvl, fitted_row)
                if (
                    postfit_diagnostics is not None
                    and irow % POSTFIT_DIAGNOSTIC_STRIDE == 0
                    and np.any(np.isfinite(before_postfit))
                    and np.any(np.isfinite(fitted_row))
                ):
                    with np.errstate(invalid='ignore'):
                        max_abs_change = np.nanmax(np.abs(fitted_row - before_postfit))
                    if np.isfinite(max_abs_change) and max_abs_change > 1e-6:
                        time_value = np.nan
                        if time_1s is not None and irow < len(time_1s):
                            time_value = float(time_1s[irow])
                        postfit_diagnostics.append({
                            'row': irow,
                            'time': time_value,
                            'target_iter': target_iter,
                            'before': before_postfit,
                            'after': fitted_row.copy(),
                            'max_abs_change': float(max_abs_change),
                        })
            fitted[irow] = fitted_row
        except Exception:
            fitted[irow] = row
    return np.clip(fitted, 0.0, 1.0)


def infer_final_iter(albedo_native, alb_iterations):
    """Infer the final iteration by matching final.dat to numbered iteration files."""
    final_albedo = np.asarray(albedo_native.get('final', []), dtype=float)
    if final_albedo.size == 0 or np.all(~np.isfinite(final_albedo)):
        return None
    best_iter = None
    best_diff = np.inf
    for iter_num, iter_albedo in alb_iterations.items():
        iter_albedo = np.asarray(iter_albedo, dtype=float)
        valid = np.isfinite(final_albedo) & np.isfinite(iter_albedo)
        if np.count_nonzero(valid) == 0:
            continue
        diff = np.nanmean(np.abs(final_albedo[valid] - iter_albedo[valid]))
        if diff < best_diff:
            best_diff = diff
            best_iter = iter_num
    return best_iter


def extend_final_albedo_1s(native_wvl, final_1s, leg_native_final, extension_wvl, leg_extension, clear_sky):
    """Extend true 1s native albedo using final_extension.dat as the preferred template."""
    native_wvl = np.asarray(native_wvl, dtype=float)
    final_1s = np.asarray(final_1s, dtype=float)
    leg_native_final = np.asarray(leg_native_final, dtype=float)
    extension_wvl = np.asarray(extension_wvl, dtype=float)
    leg_extension = np.asarray(leg_extension, dtype=float)

    use_template = (
        extension_wvl.size > 0
        and leg_extension.size == extension_wvl.size
        and np.any(np.isfinite(leg_extension))
        and leg_native_final.size == native_wvl.size
        and np.any(np.isfinite(leg_native_final))
    )

    if not use_template:
        ext_wvl = None
        extended_rows = []
        for row in final_1s:
            if np.all(~np.isfinite(row)):
                if ext_wvl is None:
                    ext_wvl, _ = alb_extention(native_wvl, np.zeros_like(native_wvl), clear_sky=clear_sky)
                extended_rows.append(np.full(ext_wvl.shape, np.nan, dtype=float))
                continue
            row_ext_wvl, row_ext = alb_extention(native_wvl, row, clear_sky=clear_sky)
            if ext_wvl is None:
                ext_wvl = row_ext_wvl
            extended_rows.append(np.interp(ext_wvl, row_ext_wvl, row_ext, left=np.nan, right=np.nan))
        return ext_wvl if ext_wvl is not None else np.array([]), np.vstack(extended_rows) if extended_rows else np.empty((0, 0))

    extended = np.full((final_1s.shape[0], extension_wvl.size), np.nan, dtype=float)
    native_range = (extension_wvl >= np.nanmin(native_wvl)) & (extension_wvl <= np.nanmax(native_wvl))
    for irow, row in enumerate(final_1s):
        valid_ratio = (
            np.isfinite(row)
            & np.isfinite(leg_native_final)
            & (np.abs(leg_native_final) > 1e-6)
        )
        if np.count_nonzero(valid_ratio) < 2:
            row_ext = leg_extension.copy()
            finite_row = np.isfinite(row)
            if np.any(native_range) and np.count_nonzero(finite_row) >= 2:
                row_ext[native_range] = np.interp(
                    extension_wvl[native_range],
                    native_wvl[finite_row],
                    row[finite_row],
                    left=np.nan,
                    right=np.nan,
                )
            extended[irow] = np.clip(row_ext, 0.0, 1.0)
            continue

        ratio_native = row[valid_ratio] / leg_native_final[valid_ratio]
        ratio_ext = np.interp(
            extension_wvl,
            native_wvl[valid_ratio],
            ratio_native,
            left=ratio_native[0],
            right=ratio_native[-1],
        )
        row_ext = leg_extension * ratio_ext
        finite_row = np.isfinite(row)
        if np.any(native_range) and np.count_nonzero(finite_row) >= 2:
            row_ext[native_range] = np.interp(
                extension_wvl[native_range],
                native_wvl[finite_row],
                row[finite_row],
                left=np.nan,
                right=np.nan,
            )
        extended[irow] = np.clip(row_ext, 0.0, 1.0)
    return extension_wvl, extended


def build_record_albedo_1s(record, fdir_lrt, stem_time, native_wvl, clear_sky, date_s=None):
    """Build true 1s iter1, iter2, final, and final-extension albedo for one record."""
    fup = np.asarray(record['ssfr_fup'], dtype=float)
    fdn = np.asarray(record['ssfr_fdn'], dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        raw_1s = np.divide(fup, fdn, out=np.full_like(fup, np.nan, dtype=float), where=fdn != 0)
    raw_1s = np.clip(raw_1s, 0.0, 1.0)

    final_iter = record['final_iter']
    if final_iter is None or not np.isfinite(final_iter):
        final_iter = infer_final_iter(
            {
                'final': record['alb_final'],
            },
            record['alb_iterations'],
        )
    final_iter = int(final_iter) if final_iter is not None and np.isfinite(final_iter) else 2
    max_target_iter = max(2, final_iter)

    current = raw_1s
    iter1_1s = np.repeat(np.asarray(record['alb_corrected'], dtype=float)[np.newaxis, :], raw_1s.shape[0], axis=0)
    iter2_1s = np.repeat(np.asarray(record['alb_fitted_baseline'], dtype=float)[np.newaxis, :], raw_1s.shape[0], axis=0)
    final_1s = np.repeat(np.asarray(record['alb_final'], dtype=float)[np.newaxis, :], raw_1s.shape[0], axis=0)
    corr0_1s = np.full_like(raw_1s, np.nan, dtype=float)
    postfit_diagnostics = []

    for target_iter in range(1, max_target_iter + 1):
        corr_iter = target_iter - 1
        corr_csv = f'{fdir_lrt}/ssfr_simu_flux_{stem_time}_iteration_{corr_iter}.csv'
        corr_factor = read_iteration_corr_factor(corr_csv, native_wvl)
        if corr_factor is None:
            if target_iter == 2:
                current = fit_albedo_1s(
                    native_wvl,
                    current,
                    record['alt'],
                    clear_sky,
                    date_s=date_s,
                    postfit_diagnostics=postfit_diagnostics,
                    time_1s=record['time'],
                    target_iter=target_iter,
                )
                iter2_1s = current
            break

        corrected = apply_corr_factor_1s(current, corr_factor)
        if target_iter == 1:
            iter1_1s = corrected
            corr0_1s = np.broadcast_to(corr_factor, corrected.shape).copy()
            current = corrected
        else:
            current = fit_albedo_1s(
                native_wvl,
                corrected,
                record['alt'],
                clear_sky,
                date_s=date_s,
                postfit_diagnostics=postfit_diagnostics,
                time_1s=record['time'],
                target_iter=target_iter,
            )
            if target_iter == 2:
                iter2_1s = current
        if target_iter == final_iter:
            final_1s = current

    if final_iter == 1:
        final_1s = iter1_1s
    elif final_iter == 2:
        final_1s = iter2_1s

    ext_wvl_1s, final_ext_1s = extend_final_albedo_1s(
        native_wvl,
        final_1s,
        record['alb_final'],
        record['extension_wvl'],
        record['alb_final_extension'],
        clear_sky,
    )
    return {
        'raw_alb_all_1s': raw_1s,
        'correction_factor_iter0_all_1s': corr0_1s,
        'alb_iter1_all_1s': iter1_1s,
        'alb_iter2_all_1s': iter2_1s,
        'alb_final_all_1s': final_1s,
        'extension_wvl_1s': ext_wvl_1s,
        'alb_final_ext_all_1s': final_ext_1s,
        'postfit_diagnostics': postfit_diagnostics,
    }


def collect_iteration_albedos(fdir_alb, stem_alb, native_wvl):
    """Read all numbered native-grid iteration albedo files for one leg."""
    iterations = {}
    iteration_files = {}
    pattern = f'sfc_alb_{stem_alb}_iter_*.dat'
    for path in Path(fdir_alb).glob(pattern):
        parts = path.stem.rsplit('_iter_', 1)
        if len(parts) != 2:
            continue
        try:
            iter_num = int(parts[1])
        except ValueError:
            continue
        alb_wvl, alb = maybe_read_two_column_file(str(path), 'alb')
        iterations[iter_num] = native_albedo_values(alb_wvl, alb, native_wvl)
        iteration_files[iter_num] = str(path)

    return (
        dict(sorted(iterations.items())),
        dict(sorted(iteration_files.items())),
    )


def all_nonfinite(values):
    """Return True when an array has no finite values."""
    values = np.asarray(values, dtype=float)
    return values.size == 0 or np.all(~np.isfinite(values))


def final_iteration_from_extension(df_extension):
    """Return final iteration encoded in a final-extension CSV, if present."""
    if df_extension is None or 'final_iter' not in df_extension:
        return None
    values = pd.to_numeric(df_extension['final_iter'], errors='coerce').dropna().unique()
    if len(values) == 0:
        return None
    return int(round(float(values[0])))


def plot_gas_absorption_bands(ax, wvl=None, alt=None):
    """Shade configured gas absorption bands on an albedo axis."""
    if wvl is not None and alt is not None:
        wvl = np.asarray(wvl, dtype=float)
        masked = gas_abs_masking(wvl, np.zeros_like(wvl, dtype=float), alt=alt)
        ax.fill_between(
            wvl,
            -0.05,
            1.05,
            where=np.isnan(masked),
            color='gray',
            alpha=0.3,
        )
        return

    for band_start, band_end in gas_bands:
        ax.axvspan(band_start, band_end, color='gray', alpha=0.3)


def plot_native_albedo_diagnostic(record, series_keys, labels, colors, filename, title):
    """Plot one native-grid leg diagnostic from stored albedo spectra."""
    import matplotlib.pyplot as plt

    wvl = np.asarray(record['native_wvl'], dtype=float)
    fig, ax = plt.subplots(figsize=(9, 5))
    for key, label, color in zip(series_keys, labels, colors):
        values = np.asarray(record[key], dtype=float)
        if values.size != wvl.size or np.all(~np.isfinite(values)):
            continue
        ax.plot(wvl, values, '-', color=color, label=label)

    plot_gas_absorption_bands(ax, wvl=wvl, alt=record.get('alt_avg'))
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('Surface Albedo', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(350, 2000)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close(fig)


def plot_all_iteration_albedo_diagnostic(record, filename, title):
    """Plot all numbered iteration albedo spectra for one leg."""
    import matplotlib.pyplot as plt

    iterations = record.get('alb_iterations', {})
    if not iterations:
        return None

    wvl = np.asarray(record['native_wvl'], dtype=float)
    finite_items = []
    for iter_num, values in iterations.items():
        values = np.asarray(values, dtype=float)
        if values.size == wvl.size and np.any(np.isfinite(values)):
            finite_items.append((iter_num, values))
    if not finite_items:
        return None

    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap('viridis')
    denom = max(len(finite_items) - 1, 1)
    for i, (iter_num, values) in enumerate(finite_items):
        ax.plot(
            wvl,
            values,
            '-',
            color=cmap(i / denom),
            label=f'iter {iter_num}',
        )

    plot_gas_absorption_bands(ax, wvl=wvl, alt=record.get('alt_avg'))
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('Surface Albedo', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(350, 2000)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return filename


def plot_postfit_diagnostics(record, output_dir, stem, title_base):
    """Plot sampled before/after post-fit spectra for one leg."""
    import matplotlib.pyplot as plt

    diagnostics = record.get('postfit_diagnostics', [])
    if not diagnostics:
        return []

    os.makedirs(output_dir, exist_ok=True)
    wvl = np.asarray(record['native_wvl'], dtype=float)
    plot_files = []
    for diag in diagnostics:
        before = np.asarray(diag.get('before'), dtype=float)
        after = np.asarray(diag.get('after'), dtype=float)
        if before.size != wvl.size or after.size != wvl.size:
            continue
        if np.all(~np.isfinite(before)) or np.all(~np.isfinite(after)):
            continue

        row = int(diag.get('row', -1))
        target_iter = diag.get('target_iter')
        time_value = diag.get('time', np.nan)
        max_abs_change = diag.get('max_abs_change', np.nan)

        fig, (ax, ax_diff) = plt.subplots(
            2,
            1,
            figsize=(9, 6),
            sharex=True,
            gridspec_kw={'height_ratios': [3, 1]},
        )
        ax.plot(wvl, before, '-', color='black', linewidth=1.3, label='before post-fit')
        ax.plot(wvl, after, '-', color='red', linewidth=1.1, label='after post-fit')
        ax.axvspan(h2o_6_start, h2o_6_end, color='gray', alpha=0.2, label='H2O-6 refit band')
        ax.axvspan(1650, 1710, color='orange', alpha=0.14, label='1650-1710 check')
        ax.set_ylabel('Surface Albedo', fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=9, loc='center left', bbox_to_anchor=(1.02, 0.5))

        diff = after - before
        ax_diff.plot(wvl, diff, '-', color='purple', linewidth=1.0)
        ax_diff.axhline(0.0, color='gray', linestyle='--', linewidth=0.8)
        ax_diff.axvspan(h2o_6_start, h2o_6_end, color='gray', alpha=0.2)
        ax_diff.axvspan(1650, 1710, color='orange', alpha=0.14)
        ax_diff.set_xlabel('Wavelength (nm)', fontsize=12)
        ax_diff.set_ylabel('After - before', fontsize=11)
        ax_diff.set_xlim(1200, 1850)

        subtitle = f'row {row}, iter {target_iter}, max change {max_abs_change:.4f}'
        if np.isfinite(time_value):
            subtitle = f'time {time_value:.4f}, {subtitle}'
        ax.set_title(f'{title_base}\nPost-fit diagnostic: {subtitle}', fontsize=12)
        fig.tight_layout()

        filename = f'{output_dir}/{stem}_postfit_iter-{target_iter}_row-{row:05d}.png'
        fig.savefig(filename, bbox_inches='tight', dpi=150)
        plt.close(fig)
        plot_files.append(filename)

    return plot_files


def plot_mean_final_albedo(date_s, case_tag, output, fig_dir):
    """Plot mean final native-grid albedo across all processed legs."""
    import matplotlib.pyplot as plt

    wvl = np.asarray(output['native_wvl'], dtype=float)
    alb_final = np.asarray(output['alb_final'], dtype=float)
    if alb_final.ndim != 2 or alb_final.shape[1] != wvl.size:
        return None

    os.makedirs(fig_dir, exist_ok=True)
    alb_avg = np.nanmean(alb_final, axis=0)
    alb_std = np.nanstd(alb_final, axis=0)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(wvl, alb_avg, '-', color='blue', label='Mean albedo (final)')
    ax.fill_between(wvl, alb_avg - alb_std, alb_avg + alb_std, color='blue', alpha=0.1)
    plot_gas_absorption_bands(ax)
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('Surface Albedo', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(350, 2000)
    ax.set_title(f'Surface Albedo (final) for {date_s} {case_tag}', fontsize=13)
    fig.tight_layout()

    filename = f'{fig_dir}/arcsix_albedo_{date_s}_{case_tag}.png'
    fig.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return filename


def plot_broadband_albedo_map(date_s, case_tag, output, fig_dir):
    """Plot broadband final albedo by leg location, with Cartesian fallback."""
    import matplotlib.pyplot as plt

    broadband_albedo_color_limits = (0.1, 0.9)
    lon_avg = np.asarray(output['lon_avg'], dtype=float)
    lat_avg = np.asarray(output['lat_avg'], dtype=float)
    lon_all = np.asarray(output['lon_all'], dtype=float)
    lat_all = np.asarray(output['lat_all'], dtype=float)
    color_vals = np.asarray(output['broadband_alb_final'], dtype=float)
    if lon_avg.size == 0 or lat_avg.size == 0:
        return None

    os.makedirs(fig_dir, exist_ok=True)
    try:
        import cartopy
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        plt.close('all')
        central_lon = float(np.nanmean(lon_avg)) if lon_avg.size > 0 else 0.0
        proj = ccrs.NorthPolarStereo(central_longitude=central_lon)
        fig = plt.figure(figsize=(8, 8))
        ax = plt.subplot(1, 1, 1, projection=proj)
        ax.coastlines(resolution='50m', linewidth=0.8)
        ax.add_feature(cfeature.LAND, facecolor='#f0f0f0', zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor='#e6f2ff', zorder=0)

        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.6, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size': 10}

        if np.all(~np.isfinite(color_vals)):
            ax.scatter(
                lon_avg,
                lat_avg,
                s=40,
                c='red',
                transform=ccrs.PlateCarree(),
                zorder=3,
                edgecolor='k',
                label='Flight legs',
            )
        else:
            sc = ax.scatter(
                lon_avg,
                lat_avg,
                s=40,
                c=color_vals,
                cmap='viridis',
                transform=ccrs.PlateCarree(),
                zorder=3,
                edgecolor='k',
                vmin=broadband_albedo_color_limits[0],
                vmax=broadband_albedo_color_limits[1],
            )
            cbar = fig.colorbar(sc, ax=ax, orientation='vertical', pad=0.02, shrink=0.7)
            cbar.set_label('Broadband Albedo (final)', fontsize=10)

        if lon_all.size > 0 and lat_all.size > 0:
            ax.scatter(lon_all, lat_all, s=6, c='gray', alpha=0.5, transform=ccrs.PlateCarree(), zorder=2)
            lon_min, lon_max = np.nanmin(lon_all), np.nanmax(lon_all)
            lat_min, lat_max = np.nanmin(lat_all), np.nanmax(lat_all)
            pad_lon = max(0.5, (lon_max - lon_min) * 0.2)
            pad_lat = max(0.5, (lat_max - lat_min) * 0.2)
            ax.set_extent([lon_min - pad_lon, lon_max + pad_lon, lat_min - pad_lat, lat_max + pad_lat], crs=ccrs.PlateCarree())

        ax.set_title(f'Polar projection (North) - Flight legs {date_s} {case_tag}', fontsize=12)
        fig.tight_layout()
        filename = f'{fig_dir}/{date_s}_{case_tag}_broadband_albedo_vs_longitude_polar_projection.png'
        fig.savefig(filename, bbox_inches='tight', dpi=150)
        plt.close(fig)
        return filename
    except Exception as err:
        print(f'Cartopy polar plot failed for {date_s} {case_tag}; falling back to Cartesian scatter. Error: {err}')

    plt.close('all')
    fig, ax = plt.subplots(figsize=(8, 6))
    if np.all(~np.isfinite(color_vals)):
        ax.scatter(lon_avg, lat_avg, c='red', s=40, edgecolor='k')
    else:
        sc = ax.scatter(lon_avg, lat_avg, c=color_vals, cmap='viridis', s=40, edgecolor='k')
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label('Broadband Albedo (final)', fontsize=12)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Flight legs {date_s} {case_tag}')
    fig.tight_layout()
    filename = f'{fig_dir}/{date_s}_{case_tag}_broadband_albedo_vs_longitude_cartesian.png'
    fig.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return filename


def plot_processing_outputs(date_s, case_tag, records, output, fig_dir='fig', plot_every=1):
    """Generate legacy-style post-processing plots for one case."""
    plot_files = []
    leg_fig_dir = f'{fig_dir}/{date_s}'
    summary_fig_dir = f'{fig_dir}/sfc_alb_corr_lonlat'
    os.makedirs(leg_fig_dir, exist_ok=True)

    for i, record in enumerate(records):
        if plot_every and i % plot_every != 0:
            continue
        stem = (
            f'arcsix_albedo_{date_s}_{case_tag}_'
            f'{record["time_start"]:.3f}-{record["time_end"]:.3f}_'
            f'{record["alt_avg"]:.2f}km_leg-{record["leg_index"]:03d}'
        )
        title_base = (
            f'Surface Albedo {date_s} {case_tag} '
            f'time {record["time_start"]:.3f}-{record["time_end"]:.3f}, '
            f'alt {record["alt_avg"]:.2f} km'
        )

        iter012_file = f'{leg_fig_dir}/{stem}_iter0_iter1_iter2.png'
        plot_native_albedo_diagnostic(
            record,
            ('alb_initial', 'alb_corrected', 'alb_fitted_baseline'),
            ('iter 0', 'iter 1', 'iter 2'),
            ('black', 'blue', 'green'),
            iter012_file,
            f'{title_base}\niter 0, iter 1, iter 2',
        )
        plot_files.append(iter012_file)

        iter01final_file = f'{leg_fig_dir}/{stem}_iter0_iter1_final.png'
        plot_native_albedo_diagnostic(
            record,
            ('alb_initial', 'alb_corrected', 'alb_final'),
            ('iter 0', 'iter 1', 'final'),
            ('black', 'blue', 'red'),
            iter01final_file,
            f'{title_base}\niter 0, iter 1, final',
        )
        plot_files.append(iter01final_file)

        all_iter_file = f'{leg_fig_dir}/{stem}_all_iterations.png'
        all_iter_file = plot_all_iteration_albedo_diagnostic(
            record,
            all_iter_file,
            f'{title_base}\nall numbered iterations',
        )
        if all_iter_file is not None:
            plot_files.append(all_iter_file)

        postfit_files = plot_postfit_diagnostics(
            record,
            f'{leg_fig_dir}/postfit_diagnostics',
            stem,
            title_base,
        )
        plot_files.extend(postfit_files)

    mean_file = plot_mean_final_albedo(date_s, case_tag, output, summary_fig_dir)
    if mean_file is not None:
        plot_files.append(mean_file)

    map_file = plot_broadband_albedo_map(date_s, case_tag, output, summary_fig_dir)
    if map_file is not None:
        plot_files.append(map_file)

    return plot_files


def process_atm_corr_case(
    date=datetime.datetime(2024, 5, 31),
    tmhr_ranges_select=None,
    case_tag='default',
    config: Optional[FlightConfig] = None,
    simulation_interval=None,
    clear_sky=True,
    output_dir=None,
    make_plots=True,
    fig_dir='fig',
    plot_every=1,
):
    """Collect final atmospheric-correction products for one case."""
    try:
        from .setup import load_cloud_observation_legs, split_tmhr_ranges
    except ImportError:
        from setup import load_cloud_observation_legs, split_tmhr_ranges

    if tmhr_ranges_select is None:
        tmhr_ranges_select = [[14.10, 14.27]]
    if config is None:
        config = make_default_config()
    if output_dir is None:
        output_dir = f'{_fdir_general_}/sfc_alb_combined_smooth_450nm'

    date_s = date.strftime("%Y%m%d")
    tmhr_ranges_select = split_tmhr_ranges(tmhr_ranges_select, simulation_interval)
    sky_tag = 'clear' if clear_sky else 'sat_cloud'
    fdir_lrt = f'{_fdir_general_}/lrt/{date_s}_{case_tag}_{sky_tag}'
    fdir_alb = f'{_fdir_general_}/sfc_alb'

    cld_legs = load_cloud_observation_legs(
        _fdir_general_,
        config.mission,
        config.platform,
        date_s,
        case_tag,
        tmhr_ranges_select,
    )

    records = []
    skipped = []
    for ileg, cld_leg in enumerate(cld_legs):
        if cld_leg.get('skip'):
            skipped.append((ileg, f'skip-flagged: {cld_leg.get("skip_reason", "")}'))
            continue
        time_start, time_end = cld_leg['time'][0], cld_leg['time'][-1]
        alt_avg = np.round(np.nanmean(cld_leg['alt']), 2)
        stem_time = f'{date_s}_{time_start:.3f}-{time_end:.3f}_alt-{alt_avg:.2f}km'
        stem_alb = f'{date_s}_{time_start:.3f}_{time_end:.3f}_{alt_avg:.2f}km'

        final_csv = f'{fdir_lrt}/ssfr_simu_flux_{stem_time}_final.csv'
        final_extension_csv = f'{fdir_lrt}/ssfr_simu_flux_{stem_time}_final_extension.csv'
        if not os.path.exists(final_csv):
            skipped.append((ileg, final_csv))
            continue

        df_final = pd.read_csv(final_csv)
        native_wvl = df_final['wvl'].to_numpy()
        native_weight = dataframe_weight(
            df_final,
            ('simu_fdn_sfc_mean', 'simu_fdn_sfc_final', 'toa_mean', 'simu_fdn_toa_mean'),
            len(native_wvl),
        )

        alb_paths = {
            'initial': f'{fdir_alb}/sfc_alb_{stem_alb}_iter_0.dat',
            'corrected': f'{fdir_alb}/sfc_alb_{stem_alb}_iter_1.dat',
            'fitted_baseline': f'{fdir_alb}/sfc_alb_{stem_alb}_iter_2.dat',
            'final': f'{fdir_alb}/sfc_alb_{stem_alb}_final.dat',
            'final_extension': f'{fdir_alb}/sfc_alb_{stem_alb}_final_extension.dat',
        }
        albedo_native = {}
        broadband_native = {}
        for key in ('initial', 'corrected', 'fitted_baseline', 'final'):
            alb_wvl, alb = maybe_read_two_column_file(alb_paths[key], 'alb')
            albedo_native[key] = native_albedo_values(alb_wvl, alb, native_wvl)
            broadband_native[key] = weighted_broadband_albedo(albedo_native[key], native_weight, native_wvl)
        alb_iterations, alb_iteration_files = collect_iteration_albedos(fdir_alb, stem_alb, native_wvl)

        extension_wvl = np.array([])
        albedo_extension = np.array([])
        extension_weight = np.array([])
        broadband_extension = np.nan
        df_extension = None
        if os.path.exists(final_extension_csv):
            df_extension = pd.read_csv(final_extension_csv)
            ext_alb_wvl, ext_alb = maybe_read_two_column_file(alb_paths['final_extension'], 'alb')
            if ext_alb_wvl is not None:
                extension_wvl = ext_alb_wvl
                albedo_extension = ext_alb
                extension_weight = dataframe_weight(
                    df_extension,
                    ('simu_fdn_sfc_final', 'simu_fdn_sfc_mean', 'simu_fdn_toa_final'),
                    len(extension_wvl),
                )
                broadband_extension = weighted_broadband_albedo(albedo_extension, extension_weight, extension_wvl)
            elif {'wvl', 'sfc_alb_final'}.issubset(df_extension.columns):
                extension_wvl = df_extension['wvl'].to_numpy()
                albedo_extension = df_extension['sfc_alb_final'].to_numpy()
                extension_weight = dataframe_weight(
                    df_extension,
                    ('simu_fdn_sfc_final', 'simu_fdn_sfc_mean', 'simu_fdn_toa_final'),
                    len(extension_wvl),
                )
                broadband_extension = weighted_broadband_albedo(albedo_extension, extension_weight, extension_wvl)

        final_iter = final_iteration_from_extension(df_extension)
        final_albedo_source = alb_paths['final']
        final_albedo_warning = None
        fallback_albedo = None
        if all_nonfinite(albedo_native['final']):
            if final_iter is not None:
                final_iter_path = f'{fdir_alb}/sfc_alb_{stem_alb}_iter_{final_iter}.dat'
                fallback_wvl, fallback_alb = maybe_read_two_column_file(final_iter_path, 'alb')
                if fallback_wvl is not None:
                    fallback_albedo = native_albedo_values(fallback_wvl, fallback_alb, native_wvl)
                    final_albedo_source = final_iter_path
                    final_albedo_warning = (
                        f'Missing {alb_paths["final"]}; using final_iter={final_iter} '
                        'from final-extension CSV. Replace with real final albedo when available.'
                    )
                    print(f'WARNING: {final_albedo_warning}')
            if fallback_albedo is None and not all_nonfinite(albedo_extension):
                fallback_albedo = np.interp(native_wvl, extension_wvl, albedo_extension, left=np.nan, right=np.nan)
                final_albedo_source = final_extension_csv
                final_albedo_warning = (
                    f'Missing {alb_paths["final"]}; interpolated sfc_alb_final from '
                    'final-extension CSV. Replace with real native final albedo when available.'
                )
                print(f'WARNING: {final_albedo_warning}')
        if fallback_albedo is not None:
            albedo_native['final'] = fallback_albedo
            broadband_native['final'] = weighted_broadband_albedo(albedo_native['final'], native_weight, native_wvl)

        simulated_native = {
            'simu_fdn_sfc_native': dataframe_column(df_final, 'simu_fdn_sfc_mean', len(native_wvl)),
            'simu_fup_sfc_native': dataframe_column(df_final, 'simu_fup_sfc_mean', len(native_wvl)),
            'simu_fdn_aircraft_native': dataframe_column(df_final, 'simu_fdn_mean', len(native_wvl)),
            'simu_fup_aircraft_native': dataframe_column(df_final, 'simu_fup_mean', len(native_wvl)),
        }
        simulated_extension = {
            'simu_fdn_sfc_ext': dataframe_column(df_extension, 'simu_fdn_sfc_final', len(extension_wvl)),
            'simu_fup_sfc_ext': dataframe_column(df_extension, 'simu_fup_sfc_final', len(extension_wvl)),
            'simu_fdn_aircraft_ext': dataframe_column(df_extension, 'simu_fdn_p3_final', len(extension_wvl)),
            'simu_fup_aircraft_ext': dataframe_column(df_extension, 'simu_fup_p3_final', len(extension_wvl)),
        }

        hsr1_wvl = np.asarray(cld_leg.get('hsr1_wvl', np.array([])), dtype=float)
        hsr1_tot = np.asarray(cld_leg.get('hsr1_tot', np.empty((len(cld_leg['time']), 0))), dtype=float)
        hsr1_dif = np.asarray(cld_leg.get('hsr1_dif', np.empty((len(cld_leg['time']), 0))), dtype=float)
        if hsr1_tot.shape == hsr1_dif.shape and hsr1_tot.ndim == 2:
            hsr1_diffuse_ratio = np.divide(
                hsr1_dif,
                hsr1_tot,
                out=np.full(hsr1_tot.shape, np.nan, dtype=float),
                where=np.isfinite(hsr1_tot) & (hsr1_tot != 0),
            )
        else:
            hsr1_diffuse_ratio = np.empty((len(cld_leg['time']), 0), dtype=float)

        record = {
            'leg_index': ileg,
            'final_iter': final_iter,
            'final_albedo_source': final_albedo_source,
            'final_albedo_warning': final_albedo_warning,
            'time': np.asarray(cld_leg['time']),
            'time_start': time_start,
            'time_end': time_end,
            'alt': np.asarray(cld_leg['alt']),
            'alt_avg': alt_avg,
            'lon': np.asarray(cld_leg['lon']),
            'lon_avg': np.nanmean(cld_leg['lon']),
            'lat': np.asarray(cld_leg['lat']),
            'lat_avg': np.nanmean(cld_leg['lat']),
            'sza': np.asarray(cld_leg['sza']),
            'saa': np.asarray(cld_leg['saa']),
            'kt19_sfc_T': np.asarray(
                cld_leg.get('kt19_sfc_T', np.full_like(cld_leg['time'], np.nan, dtype=float))
            ),
            'ssfr_fdn': np.asarray(cld_leg['ssfr_zen']),
            'ssfr_fup': np.asarray(cld_leg['ssfr_nad']),
            'ssfr_toa': np.asarray(cld_leg['ssfr_toa']),
            'ssfr_icing': np.asarray(cld_leg.get('ssfr_icing', np.zeros(len(cld_leg['time']), dtype=bool))),
            'ssfr_icing_pre': np.asarray(cld_leg.get('ssfr_icing_pre', np.zeros(len(cld_leg['time']), dtype=bool))),
            'native_wvl': native_wvl,
            'alb_initial': albedo_native['initial'],
            'alb_corrected': albedo_native['corrected'],
            'alb_fitted_baseline': albedo_native['fitted_baseline'],
            'alb_final': albedo_native['final'],
            'alb_iterations': alb_iterations,
            'broadband_alb_initial': broadband_native['initial'],
            'broadband_alb_corrected': broadband_native['corrected'],
            'broadband_alb_fitted_baseline': broadband_native['fitted_baseline'],
            'broadband_alb_final': broadband_native['final'],
            'extension_wvl': extension_wvl,
            'alb_final_extension': albedo_extension,
            'extension_weight': extension_weight,
            'broadband_alb_final_extension': broadband_extension,
            'hsr1_wvl': hsr1_wvl,
            'hsr1_diffuse_ratio': hsr1_diffuse_ratio,
            'final_csv': final_csv,
            'final_extension_csv': final_extension_csv if os.path.exists(final_extension_csv) else None,
            'albedo_files': alb_paths,
            'albedo_iteration_files': alb_iteration_files,
        }
        record.update(simulated_native)
        record.update(simulated_extension)

        record.update(build_record_albedo_1s(record, fdir_lrt, stem_time, native_wvl, clear_sky, date_s=date_s))
        gas_mask = np.isfinite(gas_abs_masking(native_wvl, np.ones_like(native_wvl, dtype=float), alt=1))
        record['broadband_alb_iter1_all_1s'] = weighted_broadband_albedo_rows(
            record['alb_iter1_all_1s'],
            record['ssfr_toa'],
            native_wvl,
        )
        record['broadband_alb_iter2_all_1s'] = weighted_broadband_albedo_rows(
            record['alb_iter2_all_1s'],
            record['ssfr_toa'],
            native_wvl,
        )
        record['broadband_alb_final_all_1s'] = weighted_broadband_albedo_rows(
            record['alb_final_all_1s'],
            record['ssfr_toa'],
            native_wvl,
        )
        record['broadband_alb_iter1_all_filter_1s'] = weighted_broadband_albedo_rows(
            record['alb_iter1_all_1s'][:, gas_mask],
            record['ssfr_toa'][:, gas_mask],
            native_wvl[gas_mask],
        )
        record['broadband_alb_iter2_all_filter_1s'] = weighted_broadband_albedo_rows(
            record['alb_iter2_all_1s'][:, gas_mask],
            record['ssfr_toa'][:, gas_mask],
            native_wvl[gas_mask],
        )
        record['broadband_alb_final_all_filter_1s'] = weighted_broadband_albedo_rows(
            record['alb_final_all_1s'][:, gas_mask],
            record['ssfr_toa'][:, gas_mask],
            native_wvl[gas_mask],
        )
        ext_weight = record['extension_weight']
        if ext_weight.size != record['alb_final_ext_all_1s'].shape[1]:
            ext_weight = np.ones(record['alb_final_ext_all_1s'].shape[1], dtype=float)
        record['broadband_alb_final_ext_all_1s'] = weighted_broadband_albedo_rows(
            record['alb_final_ext_all_1s'],
            ext_weight,
            record['extension_wvl_1s'],
        )

        records.append(record)
        gc.collect()

    if not records:
        missing = '\n'.join(path for _, path in skipped)
        raise FileNotFoundError(f'No final atmospheric-correction products found for {date_s} {case_tag}.\n{missing}')

    native_wvl = records[0]['native_wvl']
    output = {
        'date': date_s,
        'case_tag': case_tag,
        'tmhr_ranges_select': tmhr_ranges_select,
        'native_wvl': native_wvl,
        'time_start': np.array([record['time_start'] for record in records]),
        'time_end': np.array([record['time_end'] for record in records]),
        'alt_avg': np.array([record['alt_avg'] for record in records]),
        'lon_avg': np.array([record['lon_avg'] for record in records]),
        'lat_avg': np.array([record['lat_avg'] for record in records]),
        'alb_initial': stack_or_object([record['alb_initial'] for record in records]),
        'alb_corrected': stack_or_object([record['alb_corrected'] for record in records]),
        'alb_fitted_baseline': stack_or_object([record['alb_fitted_baseline'] for record in records]),
        'alb_final': stack_or_object([record['alb_final'] for record in records]),
        'alb_iterations': np.array([record['alb_iterations'] for record in records], dtype=object),
        'broadband_alb_initial': np.array([record['broadband_alb_initial'] for record in records]),
        'broadband_alb_corrected': np.array([record['broadband_alb_corrected'] for record in records]),
        'broadband_alb_fitted_baseline': np.array([record['broadband_alb_fitted_baseline'] for record in records]),
        'broadband_alb_final': np.array([record['broadband_alb_final'] for record in records]),
        'extension_wvl': first_nonempty_array(records, 'extension_wvl'),
        'alb_final_extension': stack_or_object([record['alb_final_extension'] for record in records]),
        'broadband_alb_final_extension': np.array([record['broadband_alb_final_extension'] for record in records]),
        'extension_wvl_1s': first_nonempty_array(records, 'extension_wvl_1s'),
        'hsr1_wvl': first_nonempty_array(records, 'hsr1_wvl'),
        'final_iter': np.array([
            np.nan if record['final_iter'] is None else record['final_iter']
            for record in records
        ]),
        'final_albedo_source': np.array([record['final_albedo_source'] for record in records], dtype=object),
        'final_albedo_warnings': [
            record['final_albedo_warning']
            for record in records
            if record['final_albedo_warning'] is not None
        ],
        'albedo_iteration_files': np.array([record['albedo_iteration_files'] for record in records], dtype=object),
        'records': records,
        'skipped': skipped,
    }
    output.update({
        # Compatibility aliases for older CRE/ERA5 plotting scripts.
        'wvl': output['native_wvl'],
        'alb_iter0': output['alb_initial'],
        'alb_iter1': output['alb_corrected'],
        'alb_iter2': output['alb_final'],
        'broadband_alb_iter0': output['broadband_alb_initial'],
        'broadband_alb_iter1': output['broadband_alb_corrected'],
        'broadband_alb_iter2': output['broadband_alb_final'],
        'broadband_alb_iter0_filter': output['broadband_alb_initial'],
        'broadband_alb_iter1_filter': output['broadband_alb_corrected'],
        'broadband_alb_iter2_filter': output['broadband_alb_final'],
        'ext_wvl': output['extension_wvl'],
        'time_all': concatenate_record_arrays(records, 'time'),
        'lon_all': concatenate_record_arrays(records, 'lon'),
        'lat_all': concatenate_record_arrays(records, 'lat'),
        'alt_all': concatenate_record_arrays(records, 'alt'),
        'sza_all': concatenate_record_arrays(records, 'sza'),
        'saa_all': concatenate_record_arrays(records, 'saa'),
        'kt19_sfc_T_all': concatenate_record_arrays(records, 'kt19_sfc_T'),
        'fdn_all': concatenate_record_arrays(records, 'ssfr_fdn'),
        'fup_all': concatenate_record_arrays(records, 'ssfr_fup'),
        'toa_expand_all': concatenate_record_arrays(records, 'ssfr_toa'),
        'raw_alb_all_1s': concatenate_record_arrays(records, 'raw_alb_all_1s'),
        'correction_factor_iter0_all_1s': concatenate_record_arrays(records, 'correction_factor_iter0_all_1s'),
        'icing_all': concatenate_record_arrays(records, 'ssfr_icing'),
        'icing_pre_all': concatenate_record_arrays(records, 'ssfr_icing_pre'),
        'alb_iter1_all_1s': concatenate_record_arrays(records, 'alb_iter1_all_1s'),
        'alb_iter2_all_1s': concatenate_record_arrays(records, 'alb_iter2_all_1s'),
        'alb_final_all_1s': concatenate_record_arrays(records, 'alb_final_all_1s'),
        'broadband_alb_iter1_all_1s': concatenate_record_arrays(records, 'broadband_alb_iter1_all_1s'),
        'broadband_alb_iter2_all_1s': concatenate_record_arrays(records, 'broadband_alb_iter2_all_1s'),
        'broadband_alb_final_all_1s': concatenate_record_arrays(records, 'broadband_alb_final_all_1s'),
        'broadband_alb_iter1_all_filter_1s': concatenate_record_arrays(records, 'broadband_alb_iter1_all_filter_1s'),
        'broadband_alb_iter2_all_filter_1s': concatenate_record_arrays(records, 'broadband_alb_iter2_all_filter_1s'),
        'broadband_alb_final_all_filter_1s': concatenate_record_arrays(records, 'broadband_alb_final_all_filter_1s'),
        'alb_final_ext_all_1s': concatenate_record_arrays(records, 'alb_final_ext_all_1s'),
        'broadband_alb_final_ext_all_1s': concatenate_record_arrays(records, 'broadband_alb_final_ext_all_1s'),
        'hsr1_diffuse_ratio_all': concatenate_record_arrays(records, 'hsr1_diffuse_ratio'),
        'simu_fdn_sfc_native_all': repeat_spectral_by_time(records, 'simu_fdn_sfc_native', output['native_wvl']),
        'simu_fup_sfc_native_all': repeat_spectral_by_time(records, 'simu_fup_sfc_native', output['native_wvl']),
        'simu_fdn_aircraft_native_all': repeat_spectral_by_time(records, 'simu_fdn_aircraft_native', output['native_wvl']),
        'simu_fup_aircraft_native_all': repeat_spectral_by_time(records, 'simu_fup_aircraft_native', output['native_wvl']),
        'simu_fdn_sfc_ext_all': repeat_spectral_by_time(records, 'simu_fdn_sfc_ext', output['extension_wvl_1s']),
        'simu_fup_sfc_ext_all': repeat_spectral_by_time(records, 'simu_fup_sfc_ext', output['extension_wvl_1s']),
        'simu_fdn_aircraft_ext_all': repeat_spectral_by_time(records, 'simu_fdn_aircraft_ext', output['extension_wvl_1s']),
        'simu_fup_aircraft_ext_all': repeat_spectral_by_time(records, 'simu_fup_aircraft_ext', output['extension_wvl_1s']),
    })
    output.update({
        'alb_final_all': output['alb_final_all_1s'],
        'alb_iter1_all': output['alb_iter1_all_1s'],
        'alb_iter2_all': output['alb_final_all_1s'],
        'broadband_alb_final_all': output['broadband_alb_final_all_1s'],
        'broadband_alb_iter1_all': output['broadband_alb_iter1_all_1s'],
        'broadband_alb_iter2_all': output['broadband_alb_final_all_1s'],
        'broadband_alb_iter1_all_filter': output['broadband_alb_iter1_all_filter_1s'],
        'broadband_alb_iter2_all_filter': output['broadband_alb_final_all_filter_1s'],
        'alb_iter2_ext_all': output['alb_final_ext_all_1s'],
        'broadband_alb_iter2_ext_all': output['broadband_alb_final_ext_all_1s'],
    })

    if make_plots:
        output['plot_files'] = plot_processing_outputs(
            date_s,
            case_tag,
            records,
            output,
            fig_dir=fig_dir,
            plot_every=plot_every,
        )
    else:
        output['plot_files'] = []

    os.makedirs(output_dir, exist_ok=True)
    output_file = (
        f'{output_dir}/sfc_alb_update_{date_s}_{case_tag}_'
        f'time_{tmhr_ranges_select[0][0]:.3f}_{tmhr_ranges_select[-1][-1]:.3f}.pkl'
    )
    with open(output_file, 'wb') as f:
        pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved processed atmospheric-correction product to {output_file}")
    return output_file


def process_catalog_case(config, case_id, output_dir=None, make_plots=True, fig_dir='fig', plot_every=1):
    """Process one active atmospheric-correction catalog case by id."""
    try:
        from .case_catalog import get_case
    except ImportError:
        from case_catalog import get_case

    case = get_case(case_id)
    year, month, day = [int(part) for part in case['date'].split('-')]
    return process_atm_corr_case(
        date=datetime.datetime(year, month, day),
        tmhr_ranges_select=case['tmhr_ranges_select'],
        case_tag=case['case_tag'],
        config=config,
        simulation_interval=case['simulation_interval'],
        clear_sky=case['clear_sky'],
        output_dir=output_dir,
        make_plots=make_plots,
        fig_dir=fig_dir,
        plot_every=plot_every,
    )
