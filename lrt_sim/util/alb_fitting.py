import pickle
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter1d

_PKL_DIR = Path(__file__).resolve().parent.parent

# mpl.use('Agg')


def _array_cache_key(values):
    """Return a hashable wavelength key for repeated fixed-grid calculations."""
    return tuple(np.asarray(values, dtype=float).ravel())


def _snicar_filename(clear_sky):
    if clear_sky:
        return str(_PKL_DIR / 'snicar_model_results_direct.pkl')
    return str(_PKL_DIR / 'snicar_model_results_diffuse.pkl')


@lru_cache(maxsize=2)
def _load_snicar_library(clear_sky):
    """Load each SNICAR library once per process."""
    with open(_snicar_filename(clear_sky), 'rb') as f:
        snicar_data = pickle.load(f)

    keys = tuple(snicar_data.keys())
    model_wvls_nm = tuple(
        np.asarray(snicar_data[key]['wvl'], dtype=float) * 1000.0
        for key in keys
    )
    model_albedos = tuple(
        np.asarray(snicar_data[key]['albedo'], dtype=float).copy()
        for key in keys
    )
    return keys, model_wvls_nm, model_albedos


@lru_cache(maxsize=32)
def _interpolated_snicar_library(clear_sky, obs_wvl_key):
    """Cache the SNICAR library interpolated onto one observed wavelength grid."""
    obs_wvl = np.asarray(obs_wvl_key, dtype=float)
    keys, model_wvls_nm, model_albedos = _load_snicar_library(clear_sky)
    spectra = np.vstack([
        np.interp(obs_wvl, model_wvl, model_alb)
        for model_wvl, model_alb in zip(model_wvls_nm, model_albedos)
    ])
    return keys, spectra


def _best_fit_from_matrix(keys, spectra, obs_albedo):
    """Vectorized best-fit lookup for a pre-interpolated model matrix."""
    obs_albedo = np.asarray(obs_albedo, dtype=float)
    valid = np.isfinite(obs_albedo)
    if np.count_nonzero(valid) == 0:
        return None, None, np.inf, None

    diff = spectra[:, valid] - obs_albedo[valid]
    with np.errstate(invalid='ignore'):
        rmse = np.sqrt(np.nanmean(diff * diff, axis=1))
    finite_rmse = np.isfinite(rmse)
    if not np.any(finite_rmse):
        return None, None, np.inf, None

    best_index = int(np.nanargmin(rmse))
    return keys[best_index], spectra[best_index], float(rmse[best_index]), best_index


def _find_best_fit_cached(clear_sky, obs_wvl, obs_albedo):
    """Find the best cached SNICAR spectrum on the requested wavelength grid."""
    obs_wvl_key = _array_cache_key(obs_wvl)
    keys, spectra = _interpolated_snicar_library(clear_sky, obs_wvl_key)
    best_fit_key, best_fit_spectrum, min_rmse, best_index = _best_fit_from_matrix(
        keys,
        spectra,
        obs_albedo,
    )
    if best_index is None:
        return best_fit_key, best_fit_spectrum, min_rmse, None, None

    _, model_wvls_nm, model_albedos = _load_snicar_library(clear_sky)
    return (
        best_fit_key,
        best_fit_spectrum,
        min_rmse,
        model_wvls_nm[best_index],
        model_albedos[best_index],
    )


def _best_fit_indices_from_matrix(spectra, obs_albedo_2d, chunk_size=256):
    """Return best model indices for many observed spectra without a 3-D diff array."""
    obs_albedo_2d = np.asarray(obs_albedo_2d, dtype=float)
    spectra_sq = spectra * spectra
    best_indices = np.full(obs_albedo_2d.shape[0], -1, dtype=int)
    min_rmse = np.full(obs_albedo_2d.shape[0], np.inf, dtype=float)

    for start in range(0, obs_albedo_2d.shape[0], chunk_size):
        stop = min(start + chunk_size, obs_albedo_2d.shape[0])
        obs = obs_albedo_2d[start:stop]
        valid = np.isfinite(obs)
        counts = np.count_nonzero(valid, axis=1)
        has_valid = counts > 0
        if not np.any(has_valid):
            continue

        obs_zero = np.where(valid, obs, 0.0)
        sse = (
            valid.astype(float) @ spectra_sq.T
            + np.sum(obs_zero * obs_zero, axis=1)[:, np.newaxis]
            - 2.0 * (obs_zero @ spectra.T)
        )
        sse = np.maximum(sse, 0.0)
        rmse = np.full_like(sse, np.inf, dtype=float)
        rmse[has_valid] = np.sqrt(sse[has_valid] / counts[has_valid, np.newaxis])

        finite = np.isfinite(rmse)
        rmse[~finite] = np.inf
        chunk_best_indices = np.argmin(rmse, axis=1)
        chunk_min_rmse = rmse[np.arange(rmse.shape[0]), chunk_best_indices]
        good = np.isfinite(chunk_min_rmse)
        row_indices = np.arange(start, stop)[good]
        best_indices[row_indices] = chunk_best_indices[good]
        min_rmse[row_indices] = chunk_min_rmse[good]

    return best_indices, min_rmse


@lru_cache(maxsize=128)
def _gas_absorption_mask(wvl_key, h2o_6_end, reduced_low_altitude):
    """Cache the fixed gas-absorption mask for one wavelength grid."""
    wvl = np.asarray(wvl_key, dtype=float)
    o2a_1_start, o2a_1_end = 748, 780
    h2o_1_start, h2o_1_end = 650, 706
    h2o_2_start, h2o_2_end = 705, 760
    h2o_3_start, h2o_3_end = 884, 996
    h2o_4_start, h2o_4_end = 1084, 1175
    h2o_5_start, h2o_5_end = 1230, 1286
    h2o_6_start, h2o_6_end = 1290, h2o_6_end
    h2o_7_start, h2o_7_end = 1748, 2050
    h2o_8_start, h2o_8_end = 801, 843
    final_start, final_end = 2110, 2200

    full_mask = (
        ((wvl >= o2a_1_start) & (wvl <= o2a_1_end))
        | ((wvl >= h2o_1_start) & (wvl <= h2o_1_end))
        | ((wvl >= h2o_2_start) & (wvl <= h2o_2_end))
        | ((wvl >= h2o_3_start) & (wvl <= h2o_3_end))
        | ((wvl >= h2o_4_start) & (wvl <= h2o_4_end))
        | ((wvl >= h2o_5_start) & (wvl <= h2o_5_end))
        | ((wvl >= h2o_6_start) & (wvl <= h2o_6_end))
        | ((wvl >= h2o_7_start) & (wvl <= h2o_7_end))
        | ((wvl >= h2o_8_start) & (wvl <= h2o_8_end))
        | ((wvl >= final_start) & (wvl <= final_end))
    )

    if not reduced_low_altitude:
        return full_mask

    return (
        ((wvl >= o2a_1_start) & (wvl <= o2a_1_end))
        | ((wvl >= h2o_3_start) & (wvl <= h2o_3_end))
        | ((wvl >= h2o_4_start) & (wvl <= h2o_4_end))
        | ((wvl >= h2o_5_start) & (wvl <= h2o_5_end))
        | ((wvl >= h2o_6_start) & (wvl <= h2o_6_end))
        | ((wvl >= h2o_7_start) & (wvl <= h2o_7_end))
        | ((wvl >= final_start) & (wvl <= final_end))
    )


def _fill_nan_ffill_bfill(values, limit=2):
    """Fill NaNs with repeated limited forward/backward fill using NumPy."""
    filled = np.asarray(values, dtype=float).copy()
    if not np.any(np.isnan(filled)) or np.all(np.isnan(filled)):
        return filled

    max_iter = max(1, int(np.ceil(filled.size / max(limit, 1))) + 1)
    for _ in range(max_iter):
        n_missing_before = np.count_nonzero(np.isnan(filled))

        last = np.nan
        n_from_last = 0
        for i, value in enumerate(filled):
            if np.isfinite(value):
                last = value
                n_from_last = 0
            elif np.isfinite(last) and n_from_last < limit:
                filled[i] = last
                n_from_last += 1

        last = np.nan
        n_from_last = 0
        for i in range(filled.size - 1, -1, -1):
            value = filled[i]
            if np.isfinite(value):
                last = value
                n_from_last = 0
            elif np.isfinite(last) and n_from_last < limit:
                filled[i] = last
                n_from_last += 1

        if not np.any(np.isnan(filled)):
            break
        if np.count_nonzero(np.isnan(filled)) == n_missing_before:
            break

    return filled


@lru_cache(maxsize=64)
def _snowice_band_masks(wvl_key, h2o_6_end):
    """Cache fixed wavelength bands used by snowice_alb_fitting."""
    alb_wvl = np.asarray(wvl_key, dtype=float)
    alb_wvl_sep_1nd_s, alb_wvl_sep_1nd_e = 370, 800
    alb_wvl_sep_2nd_s, alb_wvl_sep_2nd_e = 795, 850
    alb_wvl_sep_3rd_s, alb_wvl_sep_3rd_e = 850, 1050
    alb_wvl_sep_4th_s, alb_wvl_sep_4th_e = 1050, 1210
    alb_wvl_sep_6th_s, alb_wvl_sep_6th_e = 1520, 2100
    if h2o_6_end > 1520:
        alb_wvl_sep_6th_s = h2o_6_end + 5

    return (
        (alb_wvl >= alb_wvl_sep_1nd_s) & (alb_wvl < alb_wvl_sep_1nd_e),
        (alb_wvl >= alb_wvl_sep_2nd_s) & (alb_wvl < alb_wvl_sep_2nd_e),
        (alb_wvl >= alb_wvl_sep_3rd_s) & (alb_wvl < alb_wvl_sep_3rd_e),
        (alb_wvl >= alb_wvl_sep_4th_s) & (alb_wvl < alb_wvl_sep_4th_e),
        (alb_wvl >= 1185) & (alb_wvl < 1290),
        (alb_wvl >= 1285) & (alb_wvl < 1520),
        (alb_wvl >= 1515) & (alb_wvl < 1700),
        (alb_wvl >= alb_wvl_sep_6th_s) & (alb_wvl <= alb_wvl_sep_6th_e),
    )


def gas_abs_masking(
    wvl,
    alb,
    alt,
    h2o_6_end=1509,
    interp_nan=True,
    altitude_dependent=False,
):
    """Mask all gas bands by default, with optional reduced low-altitude masking."""
    wvl = np.asarray(wvl, dtype=float)
    alb = np.asarray(alb, dtype=float)
    reduced_low_altitude = bool(altitude_dependent and alt <= 0.5)
    mask = _gas_absorption_mask(_array_cache_key(wvl), float(h2o_6_end), reduced_low_altitude)

    effective_mask_ = np.ones_like(alb)
    alb_mask = alb.copy()
    alb_mask[mask] = np.nan
    effective_mask_[mask] = np.nan
    
    # interpolation if nan in effective_mask_ range
    if interp_nan and np.sum(~np.isnan(effective_mask_)) != np.isfinite(alb_mask).sum():
        fit_wvl_mask = np.logical_and(~np.isnan(effective_mask_), np.isnan(alb_mask))
        unmasked_values = alb_mask[effective_mask_ == 1]
        unmasked_nan = np.isnan(unmasked_values)
        if np.any(unmasked_nan):
            filled = _fill_nan_ffill_bfill(unmasked_values, limit=2)
            alb_mask[fit_wvl_mask] = filled[unmasked_nan]
        
    # plt.close('all')
    # plt.plot(wvl, before_interp, '-o', label='Before fill')
    # plt.plot(wvl, alb_mask, '-x', label='After fill')
    # plt.xlabel('Wavelength (nm)')
    # plt.ylabel('Albedo')
    # plt.legend()
    # plt.show()
     
    
    return alb_mask
   



def find_best_fit(model_library, obs_wvl, obs_albedo):
    """
    Finds the best-fit model spectrum from a library by minimizing RMSE
    at *only* the provided obs_wvl points.
    """
    
    obs_wvl = np.asarray(obs_wvl, dtype=float)
    keys = tuple(model_library.keys())
    model_wvls = tuple(
        np.asarray(model_library[key]['wvl'], dtype=float) * 1000.0
        for key in keys
    )
    model_albedos = tuple(
        np.asarray(model_library[key]['albedo'], dtype=float)
        for key in keys
    )
    spectra = np.vstack([
        np.interp(obs_wvl, model_wvl, model_alb)
        for model_wvl, model_alb in zip(model_wvls, model_albedos)
    ])
    best_fit_params, best_fit_spectrum, min_rmse, best_index = _best_fit_from_matrix(
        keys,
        spectra,
        obs_albedo,
    )
    if best_index is None:
        return None, None, np.inf, None, None
    model_wvl = model_wvls[best_index]
    ori_model_spectrum = model_albedos[best_index].copy()
        
    # plt.close('all')
    # plt.plot(obs_wvl, best_fit_spectrum, '-', color='r', label='Best Fit Model')
    # plt.plot(obs_wvl, obs_albedo, '-', color='k', label='Observed')
    # plt.plot(model_wvl, ori_model_spectrum, '--', color='gray', label='Original Best Fit Model Spectrum')
    # plt.xlabel('Wavelength (nm)')
    # plt.ylabel('Albedo')
    # plt.legend()
    # plt.title(f'Best Fit Model: {best_fit_params}, RMSE: {min_rmse:.4f}')
    # plt.show()
    
        
    return best_fit_params, best_fit_spectrum, min_rmse, model_wvl, ori_model_spectrum


def _smooth_h2o6_h2o7_continuum(alb_wvl, alb, h2o_6_end, h2o_7_start=1748):
    """Remove narrow spikes and smooth the clean continuum before H2O-7."""
    window_start = h2o_6_end + 5
    window = (alb_wvl >= window_start) & (alb_wvl < h2o_7_start)
    if np.count_nonzero(window) < 5:
        return alb

    smoothed = alb.copy()
    y = smoothed[window].copy()
    finite = np.isfinite(y)
    if np.count_nonzero(finite) < 5:
        return smoothed
    if not np.all(finite):
        x = alb_wvl[window]
        y[~finite] = np.interp(x[~finite], x[finite], y[finite])

    continuum = uniform_filter1d(y, size=11, mode='reflect')
    resid = y - continuum
    mad = np.nanmedian(np.abs(resid - np.nanmedian(resid)))
    sigma = 1.4826 * mad if mad > 0 else np.nanstd(resid)

    if np.isfinite(sigma) and sigma > 0:
        spike = np.abs(resid) > 3 * sigma
        y[spike] = continuum[spike]

    y = uniform_filter1d(y, size=9, mode='reflect')
    smoothed[window] = y
    return np.clip(smoothed, 0, 1)


def _fill_h2o6_with_scaled_snicar(alb_wvl, alb_corr_fit, alb_corr_mask, best_fit_spectrum, h2o_6_end):
    """Fill H2O-6 with the best-fit SNICAR shape before generic gap bridging."""
    h2o_6_start = 1290
    h2o6_gap = (
        (alb_wvl >= h2o_6_start)
        & (alb_wvl <= h2o_6_end)
        & np.isnan(alb_corr_mask)
        & np.isfinite(best_fit_spectrum)
    )
    if not np.any(h2o6_gap):
        return alb_corr_fit, alb_corr_mask

    left_anchor = (
        (alb_wvl >= 1185)
        & (alb_wvl < h2o_6_start)
        & np.isfinite(alb_corr_fit)
        & np.isfinite(best_fit_spectrum)
    )
    right_anchor = (
        (alb_wvl > h2o_6_end)
        & (alb_wvl <= 1700)
        & np.isfinite(alb_corr_fit)
        & np.isfinite(best_fit_spectrum)
    )
    anchors = left_anchor | right_anchor

    if np.count_nonzero(anchors) >= 2:
        model = best_fit_spectrum[anchors]
        obs = alb_corr_fit[anchors]
        model_var = np.sum((model - np.mean(model)) ** 2)
        if model_var > 1e-12:
            scale = np.sum((model - np.mean(model)) * (obs - np.mean(obs))) / model_var
            offset = np.mean(obs) - scale * np.mean(model)
        else:
            scale = 1.0
            offset = np.mean(obs - model)
    elif np.count_nonzero(anchors) == 1:
        anchor = np.flatnonzero(anchors)[0]
        scale = 1.0
        offset = alb_corr_fit[anchor] - best_fit_spectrum[anchor]
    else:
        scale = 1.0
        offset = 0.0

    replacement = scale * best_fit_spectrum[h2o6_gap] + offset
    alb_corr_fit = alb_corr_fit.copy()
    alb_corr_mask = alb_corr_mask.copy()
    alb_corr_fit[h2o6_gap] = np.clip(replacement, 0, 1)
    alb_corr_mask[h2o6_gap] = alb_corr_fit[h2o6_gap]
    return alb_corr_fit, alb_corr_mask


def _snowice_alb_fitting_from_best(
    alb_wvl,
    alb_corr,
    alb_corr_mask,
    best_fit_spectrum,
    h2o_6_end=1509,
):
    alb_wvl = np.asarray(alb_wvl, dtype=float)
    alb_corr = np.asarray(alb_corr, dtype=float)
    alb_corr_mask = np.asarray(alb_corr_mask, dtype=float)
    best_fit_spectrum = np.asarray(best_fit_spectrum, dtype=float)
    
    # plt.close('all')
    # plt.figure(figsize=(8, 5))
    # plt.plot(alb_wvl, alb_corr, '-', color='k', label='Corrected Albedo')
    # plt.plot(alb_wvl, alb_corr_mask, '-', color='g', label='Masked Corrected Albedo')
    # plt.plot(alb_wvl, best_fit_spectrum, '-', color='r', label='Best Fitted Albedo')
    # plt.fill_between(alb_wvl, -0.05, 1.05, where=np.isnan(alb_corr_mask), color='gray', alpha=0.2, label='Mask Gas absorption bands')
    # plt.xlim(350, 2000)
    # plt.ylim(-0.05, 1.05)
    # plt.xlabel('Wavelength (nm)')
    # plt.ylabel('Albedo')
    # plt.legend()
    # plt.title(f'SNICAR Best Fit Model: {best_fit_key}, RMSE: {min_rmse:.4f}')
    # plt.show()
    
    
    
    (
        band_1_fit,
        band_2_fit,
        band_3_fit,
        band_4_fit,
        band_5a_fit,
        band_5b_fit,
        band_5c_fit,
        band_6_fit,
    ) = _snowice_band_masks(_array_cache_key(alb_wvl), float(h2o_6_end))
    
    alb_corr_fit = alb_corr_mask.copy()
    alb_corr_fit, alb_corr_mask = _fill_h2o6_with_scaled_snicar(
        alb_wvl,
        alb_corr_fit,
        alb_corr_mask,
        best_fit_spectrum,
        h2o_6_end,
    )
    
    # --- backup: polynomial fit for band_1 anchored at 550 nm (caused unnatural std dev gap at 550 nm) ---
    # for bands_fit in [band_1_fit]:
    #     bandfit_nan = np.isnan(alb_corr_mask[bands_fit])
    #     if bandfit_nan.sum() == 0:
    #         continue
    #     bandfit_nan_ind = np.where(bandfit_nan)[0]
    #     if bandfit_nan_ind[-1] == len(bandfit_nan)-1:
    #         bandfit_nan_ind = bandfit_nan_ind[:-1]
    #     if bandfit_nan_ind.size == 0:
    #         continue
    #     left_mean_ind_num = 5
    #     if bandfit_nan_ind[0] < left_mean_ind_num:
    #         left_mean_ind_num = bandfit_nan_ind[0]
    #     xl_origin = alb_corr_fit[bands_fit][bandfit_nan_ind[0]-left_mean_ind_num:bandfit_nan_ind[0]-1].mean()
    #     right_mean_ind_num = 5
    #     if (len(bandfit_nan) - bandfit_nan_ind[-1] - 1) < right_mean_ind_num:
    #         right_mean_ind_num = len(bandfit_nan) - bandfit_nan_ind[-1] - 1
    #     xr_origin = alb_corr_fit[bands_fit][bandfit_nan_ind[-1]+1:bandfit_nan_ind[-1]+right_mean_ind_num].mean()
    #     wvl550nm_ind = np.argmin(np.abs(alb_wvl[bands_fit][~bandfit_nan]-550))
    #     fit_2nd = np.poly1d(np.polyfit(alb_wvl[bands_fit][~bandfit_nan][wvl550nm_ind:],
    #                                     alb_corr_mask[bands_fit][~bandfit_nan][wvl550nm_ind:], 2))
    #     replace_array = fit_2nd(alb_wvl[bands_fit][bandfit_nan])
    #     alb_corr_fit_replace = alb_corr_fit[bands_fit].copy()
    #     alb_corr_fit_replace[bandfit_nan] = replace_array.copy()
    #     alb_corr_fit[bands_fit] = alb_corr_fit_replace
    # --- end backup ---

    for bands_fit in [
                      band_1_fit, band_2_fit, band_3_fit, band_4_fit,
                      band_5a_fit, band_5b_fit, band_5c_fit, band_6_fit]:
        
        bandfit_nan = np.isnan(alb_corr_mask[bands_fit])
        if bandfit_nan.sum() == 0:
            continue
        bandfit_nan_ind = np.where(bandfit_nan)[0]

        band_len = int(np.sum(bands_fit))
        has_left  = bandfit_nan_ind[0] > 0
        has_right = bandfit_nan_ind[-1] + 1 < band_len

        if not has_left and not has_right:
            # entire band is NaN — nothing to anchor against, leave for ffill/bfill cleanup
            continue

        # Exact single boundary points — guarantees zero discontinuity at gap edges
        if has_left and has_right:
            xl_obs    = alb_corr_fit[bands_fit][bandfit_nan_ind[0] - 1]
            xl_snicar = best_fit_spectrum[bands_fit][bandfit_nan_ind[0] - 1]
            xr_obs    = alb_corr_fit[bands_fit][bandfit_nan_ind[-1] + 1]
            xr_snicar = best_fit_spectrum[bands_fit][bandfit_nan_ind[-1] + 1]
            denom = xr_snicar - xl_snicar
            if abs(denom) > 1e-6:
                a = (xr_obs - xl_obs) / denom
                b = xl_obs - a * xl_snicar
                replace_array = a * best_fit_spectrum[bands_fit][bandfit_nan] + b
            else:
                # SNICAR is flat in this gap — fall back to linear interpolation between observed edges
                wvl_gap = alb_wvl[bands_fit][bandfit_nan]
                wvl_l = alb_wvl[bands_fit][bandfit_nan_ind[0] - 1]
                wvl_r = alb_wvl[bands_fit][bandfit_nan_ind[-1] + 1]
                replace_array = np.interp(wvl_gap, [wvl_l, wvl_r], [xl_obs, xr_obs])
        elif has_left:
            anchor_ind = bandfit_nan_ind[0] - 1
            anchor_obs = alb_corr_fit[bands_fit][anchor_ind]
            anchor_snicar = best_fit_spectrum[bands_fit][anchor_ind]
            valid = np.isfinite(alb_corr_fit[bands_fit]) & np.isfinite(best_fit_spectrum[bands_fit])
            snicar_valid = best_fit_spectrum[bands_fit][valid]
            obs_valid = alb_corr_fit[bands_fit][valid]
            denom = np.sum((snicar_valid - np.mean(snicar_valid))**2)
            if snicar_valid.size >= 2 and denom > 1e-12:
                slope = np.sum((snicar_valid - np.mean(snicar_valid)) * (obs_valid - np.mean(obs_valid))) / denom
                replace_array = anchor_obs + slope * (best_fit_spectrum[bands_fit][bandfit_nan] - anchor_snicar)
            else:
                replace_array = np.full(np.sum(bandfit_nan), anchor_obs)
        else:  # has_right only
            anchor_ind = bandfit_nan_ind[-1] + 1
            anchor_obs = alb_corr_fit[bands_fit][anchor_ind]
            anchor_snicar = best_fit_spectrum[bands_fit][anchor_ind]
            valid = np.isfinite(alb_corr_fit[bands_fit]) & np.isfinite(best_fit_spectrum[bands_fit])
            snicar_valid = best_fit_spectrum[bands_fit][valid]
            obs_valid = alb_corr_fit[bands_fit][valid]
            denom = np.sum((snicar_valid - np.mean(snicar_valid))**2)
            if snicar_valid.size >= 2 and denom > 1e-12:
                slope = np.sum((snicar_valid - np.mean(snicar_valid)) * (obs_valid - np.mean(obs_valid))) / denom
                replace_array = anchor_obs + slope * (best_fit_spectrum[bands_fit][bandfit_nan] - anchor_snicar)
            else:
                replace_array = np.full(np.sum(bandfit_nan), anchor_obs)

        alb_corr_fit_replace = alb_corr_fit[bands_fit].copy()
        alb_corr_fit_replace[bandfit_nan] = replace_array
        alb_corr_fit[bands_fit] = alb_corr_fit_replace
        # plt.close('all')
        # plt.plot(alb_wvl[bands_fit], alb_corr_mask[bands_fit], 'o', color='k', label='Corrected Albedo')
        # plt.plot(alb_wvl[bands_fit], alb_corr_fit_replace, '--', color='b', label='Replace')
        # plt.plot(alb_wvl[bands_fit], alb_corr_fit[bands_fit], '-', color='r', label='Fitted Albedo')
        # plt.xlabel('Wavelength (nm)')
        # plt.ylabel('Albedo')
        # plt.legend()
        # plt.show()
            
    
    alb_corr_fit = np.clip(alb_corr_fit, 0, 1)
    finite_fit = np.isfinite(alb_corr_fit)
    if not np.all(finite_fit):
        if np.any(finite_fit):
            alb_corr_fit[~finite_fit] = np.interp(
                alb_wvl[~finite_fit],
                alb_wvl[finite_fit],
                alb_corr_fit[finite_fit],
                left=alb_corr_fit[finite_fit][0],
                right=alb_corr_fit[finite_fit][-1],
            )
        else:
            alb_corr_fit = np.copy(alb_corr)
            alb_corr_fit = np.clip(alb_corr_fit, 0, 1)
    
    # plt.close('all')
    # plt.figure(figsize=(8, 5))
    # plt.plot(alb_wvl, alb_corr, '-', color='k', label='Corrected Albedo', linewidth=3)
    # plt.plot(alb_wvl, alb_corr_fit, '-', color='r', label='Fitted Albedo', linewidth=1.5)
    # plt.xlabel('Wavelength (nm)')
    # plt.ylabel('Albedo')
    # plt.fill_between(alb_wvl, -0.05, 1.05, where=np.isnan(alb_corr_mask), color='gray', alpha=0.2, label='Mask Gas absorption bands')
    # plt.legend()
    # plt.xlim(350, 2000)
    # plt.ylim(-0.05, 1.05)
    # plt.title(f'SNICAR Best Fit Model: {best_fit_key}, RMSE: {min_rmse:.4f}')
    # plt.show()
    # sys.exit()
    
    # shortwave blend: moving average below 450 nm to suppress detector noise
    # replace_sel = alb_wvl < 450
    # if np.any(replace_sel):
    #     # include context above 450 nm so the filter boundary is not at the edge of the data
    #     context_sel = alb_wvl < 510
    #     context_vals = alb_corr_fit[context_sel].copy()
    #     smoothed_context = uniform_filter1d(context_vals, size=11, mode='reflect')
    #     n_replace = int(np.sum(replace_sel))
    #     alb_corr_fit[replace_sel] = smoothed_context[:n_replace]

    # --- backup: SNICAR-blend progressively trust below 450 nm ---
    # anchor_sel = (alb_wvl >= 450) & (alb_wvl <= 550)
    # if np.any(anchor_sel) and np.any(np.isfinite(alb_corr_fit[anchor_sel])) and np.any(np.isfinite(best_fit_spectrum[anchor_sel])):
    #     offset = np.nanmean(alb_corr_fit[anchor_sel]) - np.nanmean(best_fit_spectrum[anchor_sel])
    #     snicar_anchored = np.clip(best_fit_spectrum + offset, 0.0, 1.0)
    #     replace_sel = alb_wvl < 450
    #     alpha = np.clip((alb_wvl[replace_sel] - 350.0) / (450.0 - 350.0), 0.0, 1.0)
    #     alb_corr_fit[replace_sel] = alpha * alb_corr_fit[replace_sel] + (1.0 - alpha) * snicar_anchored[replace_sel]
    # --- end backup ---
    
    alb_corr_fit = np.clip(alb_corr_fit, 0, 1)
    alb_corr_fit = _smooth_h2o6_h2o7_continuum(alb_wvl, alb_corr_fit, h2o_6_end)
    
    
    # smooth with window size of 5
    alb_corr_fit_smooth = alb_corr_fit.copy()
    alb_corr_fit_smooth = uniform_filter1d(alb_corr_fit_smooth, size=5, mode='reflect')
    alb_corr_fit_smooth = np.clip(alb_corr_fit_smooth, 0, 1)

    # print("alb_wvl shape:", alb_wvl.shape)
    # print("alb_corr shape:", alb_corr.shape)
    # print("alb_corr_mask shape:", alb_corr_mask.shape)
    # print("alb_corr_fit shape:", alb_corr_fit.shape)
    # print("alb_corr_fit_smooth shape:", alb_corr_fit_smooth.shape)
    
    return alb_corr_fit_smooth


def snowice_alb_fitting(alb_wvl, alb_corr, alt, clear_sky=False, h2o_6_end=1509):
    alb_wvl = np.asarray(alb_wvl, dtype=float)
    alb_corr = np.asarray(alb_corr, dtype=float)
    alb_corr_mask = gas_abs_masking(alb_wvl, alb_corr.copy(), alt=alt, h2o_6_end=h2o_6_end)
    _, best_fit_spectrum, _, _, _ = _find_best_fit_cached(
        clear_sky=clear_sky,
        obs_wvl=alb_wvl,
        obs_albedo=alb_corr_mask
    )
    return _snowice_alb_fitting_from_best(
        alb_wvl,
        alb_corr,
        alb_corr_mask,
        best_fit_spectrum,
        h2o_6_end=h2o_6_end,
    )


def snowice_alb_fitting_batch(
    alb_wvl,
    alb_corr_2d,
    alt,
    clear_sky=False,
    h2o_6_end=1509,
    chunk_size=256,
):
    """Apply snowice_alb_fitting to many spectra with batched SNICAR selection."""
    alb_wvl = np.asarray(alb_wvl, dtype=float)
    alb_corr_2d = np.asarray(alb_corr_2d, dtype=float)
    if alb_corr_2d.ndim == 1:
        return snowice_alb_fitting(
            alb_wvl,
            alb_corr_2d,
            alt=alt,
            clear_sky=clear_sky,
            h2o_6_end=h2o_6_end,
        )

    alt_arr = np.asarray(alt, dtype=float)
    if alt_arr.ndim == 0:
        alt_arr = np.full(alb_corr_2d.shape[0], float(alt_arr), dtype=float)

    masked = np.full_like(alb_corr_2d, np.nan, dtype=float)
    all_nonfinite = np.all(~np.isfinite(alb_corr_2d), axis=1)
    for irow in np.flatnonzero(~all_nonfinite):
        row_alt = alt_arr[irow] if irow < alt_arr.size else np.nan
        masked[irow] = gas_abs_masking(
            alb_wvl,
            alb_corr_2d[irow].copy(),
            alt=row_alt,
            h2o_6_end=h2o_6_end,
        )

    wvl_key = _array_cache_key(alb_wvl)
    _, spectra = _interpolated_snicar_library(clear_sky, wvl_key)
    best_indices, _ = _best_fit_indices_from_matrix(
        spectra,
        masked,
        chunk_size=chunk_size,
    )

    fitted = np.full_like(alb_corr_2d, np.nan, dtype=float)
    for irow in np.flatnonzero(~all_nonfinite):
        best_index = best_indices[irow]
        if best_index < 0:
            fitted[irow] = np.clip(alb_corr_2d[irow], 0, 1)
            continue
        try:
            fitted[irow] = _snowice_alb_fitting_from_best(
                alb_wvl,
                alb_corr_2d[irow],
                masked[irow],
                spectra[best_index],
                h2o_6_end=h2o_6_end,
            )
        except Exception:
            fitted[irow] = np.clip(alb_corr_2d[irow], 0, 1)

    return np.clip(fitted, 0, 1)


def _smooth_extension_tail(wvl, albedo, start_wvl=2000.0, smooth_size=101, blend_width=50.0):
    """Smooth synthetic longwave extension while preserving the anchor at start_wvl."""
    wvl = np.asarray(wvl, dtype=float)
    smoothed_albedo = np.asarray(albedo, dtype=float).copy()
    tail = wvl >= start_wvl
    if np.count_nonzero(tail) < 5:
        return smoothed_albedo

    x = wvl[tail]
    y = smoothed_albedo[tail].copy()
    finite = np.isfinite(y)
    if np.count_nonzero(finite) < 5:
        return smoothed_albedo
    if not np.all(finite):
        y[~finite] = np.interp(x[~finite], x[finite], y[finite])

    smooth_size = max(3, int(smooth_size))
    if smooth_size % 2 == 0:
        smooth_size += 1
    y_smooth = uniform_filter1d(y, size=smooth_size, mode='nearest')

    alpha = np.ones_like(y)
    if blend_width > 0:
        alpha = np.clip((x - start_wvl) / blend_width, 0.0, 1.0)
    y = (1.0 - alpha) * y + alpha * y_smooth
    smoothed_albedo[tail] = np.clip(y, 0.0, 1.0)
    return smoothed_albedo


def alb_extention(alb_wvl, alb_corr_fitted, clear_sky=False):
    # Extend albedo spectrum to 2.5 um with constant albedo at the last wavelength
    ext_wvl_start = 250
    ext_wvl_end = 4050
    ext_wvl = np.arange(ext_wvl_start, ext_wvl_end+1, 1)
    ext_alb = np.ones_like(ext_wvl) * alb_corr_fitted[-1]

    alb_wvl_ext = np.concatenate((ext_wvl[ext_wvl < alb_wvl[0]], alb_wvl, ext_wvl[ext_wvl > alb_wvl[-1]]))
    alb_corr_fitted_ext = np.concatenate((ext_alb[ext_wvl < alb_wvl[0]], alb_corr_fitted, ext_alb[ext_wvl > alb_wvl[-1]]))

    short_wvl_start, short_wvl_end = 355, 550
    long_wvl_start, long_wvl_end = 1450, 1740
    long_anchor_wvl = 1495.0
    long_blend_start = 1900.0
    long_replace_start = 2000.0
    
    # fit on long wavelength side
    long_wvl_sel = (
        (alb_wvl >= long_wvl_start)
        & (alb_wvl <= long_wvl_end)
        & np.isfinite(alb_corr_fitted)
    )
    if np.count_nonzero(long_wvl_sel) < 2:
        long_wvl_sel = (
            (alb_wvl >= 1500)
            & (alb_wvl <= 2000)
            & np.isfinite(alb_corr_fitted)
        )
    long_wvl = alb_wvl[long_wvl_sel]
    long_wvl_alb = alb_corr_fitted[long_wvl_sel]
    best_fit_key, best_fit_spectrum, min_rmse, ori_spec_wvl, ori_spec_alb = _find_best_fit_cached(
        clear_sky=clear_sky,
        obs_wvl=long_wvl,
        obs_albedo=long_wvl_alb
    )
    

    interp_ori_spec_alb = np.interp(alb_wvl_ext, ori_spec_wvl, ori_spec_alb)
    interp_snicar_unscaled = interp_ori_spec_alb.copy()

    finite_alb = np.isfinite(alb_corr_fitted)
    if np.count_nonzero(finite_alb) >= 2:
        obs_anchor = np.interp(
            long_anchor_wvl,
            alb_wvl[finite_alb],
            alb_corr_fitted[finite_alb],
        )
    elif np.count_nonzero(finite_alb) == 1:
        obs_anchor = alb_corr_fitted[finite_alb][0]
    else:
        obs_anchor = np.nan
    model_anchor = np.interp(long_anchor_wvl, ori_spec_wvl, ori_spec_alb)
    right_edge_mask = finite_alb & (alb_wvl < long_blend_start)
    if np.any(right_edge_mask):
        right_edge_wvl   = float(alb_wvl[right_edge_mask][-1])
        obs_right_edge   = float(alb_corr_fitted[right_edge_mask][-1])
        model_right_edge = float(np.interp(right_edge_wvl, ori_spec_wvl, ori_spec_alb))
        if model_right_edge > 1e-6:
            interp_ori_spec_alb = interp_ori_spec_alb * (obs_right_edge / model_right_edge)
        elif np.isfinite(obs_anchor) and np.isfinite(model_anchor):
            interp_ori_spec_alb = interp_ori_spec_alb + (obs_anchor - model_anchor)
    elif np.isfinite(obs_anchor) and np.isfinite(model_anchor):
        interp_ori_spec_alb = interp_ori_spec_alb + (obs_anchor - model_anchor)

    # Linearly rescale the post-blend/replace region (≥1900 nm) so that:
    #   ceiling: inter-band peak (≥2000 nm) is anchored via the SNICAR 1750-window ratio
    #            ceiling = obs_1750_max × (model_lm_post2000 / model_lm_1750)
    #   floor:   absorption trough is anchored to the observed minimum near 1495 nm
    #            floor = obs_anchor (observed albedo at 1495 nm)
    # The linear map raw_lo→floor, raw_hi→ceiling preserves spectral shape while
    # simultaneously respecting both observation-derived constraints. The same
    # transform is applied to the full ≥1900 nm region (blend + replace) to avoid
    # a step discontinuity at 2000 nm.
    snicar_1750_mask     = (alb_wvl_ext >= 1650) & (alb_wvl_ext <= 1900)
    snicar_post2000_mask = alb_wvl_ext >= long_replace_start
    blend_and_replace_mask = alb_wvl_ext >= long_blend_start
    obs_1750_win = (alb_wvl >= 1650) & (alb_wvl <= 1900) & np.isfinite(alb_corr_fitted)
    if (
        np.any(snicar_1750_mask)
        and np.any(snicar_post2000_mask)
        and np.count_nonzero(obs_1750_win) >= 1
        and np.isfinite(obs_anchor)
    ):
        model_lm_1750     = float(np.nanmax(interp_snicar_unscaled[snicar_1750_mask]))
        model_lm_post2000 = float(np.nanmax(interp_snicar_unscaled[snicar_post2000_mask]))
        obs_1750_max      = float(np.nanmax(alb_corr_fitted[obs_1750_win]))
        if model_lm_1750 > 1e-6 and np.isfinite(model_lm_post2000) and np.isfinite(obs_1750_max):
            ceiling   = obs_1750_max * model_lm_post2000 / model_lm_1750
            floor_val = obs_anchor
            raw_lo = float(np.nanmin(interp_ori_spec_alb[blend_and_replace_mask]))
            raw_hi = float(np.nanmax(interp_ori_spec_alb[blend_and_replace_mask]))
            if raw_hi > raw_lo and ceiling > floor_val > 0:
                scale  = (ceiling - floor_val) / (raw_hi - raw_lo)
                offset = floor_val - scale * raw_lo
                interp_ori_spec_alb[blend_and_replace_mask] = np.clip(
                    scale * interp_ori_spec_alb[blend_and_replace_mask] + offset,
                    0.0, 1.0,
                )
            elif raw_hi > ceiling > 1e-6:
                interp_ori_spec_alb[blend_and_replace_mask] *= ceiling / raw_hi

    # plt.close('all')
    # plt.figure(figsize=(8, 5))
    # plt.plot(alb_wvl, alb_corr_fitted, '-', color='gray', label='Corrected Albedo', linewidth=1)
    # plt.plot(long_wvl, long_wvl_alb, '-', color='k', label='Corrected Albedo', linewidth=3)
    # plt.plot(alb_wvl_ext[alb_wvl_ext>long_wvl_start], interp_ori_spec_alb[alb_wvl_ext>long_wvl_start], '-', color='r', label='Best Fitted Albedo', linewidth=1.5)
    # plt.xlabel('Wavelength (nm)')
    # plt.ylabel('Albedo')
    # plt.legend()
    # plt.title(f'SNICAR Best Fit Model: {best_fit_key}, RMSE: {min_rmse:.4f}')
    # plt.show()
    
    alb_corr_fitted_ext_long = np.copy(alb_corr_fitted_ext)
    long_blend_sel = (alb_wvl_ext > long_blend_start) & (alb_wvl_ext < long_replace_start)
    if np.any(long_blend_sel):
        alpha = (alb_wvl_ext[long_blend_sel] - long_blend_start) / (long_replace_start - long_blend_start)
        alpha = np.clip(alpha, 0.0, 1.0)
        alb_corr_fitted_ext_long[long_blend_sel] = (
            (1.0 - alpha) * alb_corr_fitted_ext[long_blend_sel]
            + alpha * interp_ori_spec_alb[long_blend_sel]
        )
    long_ext_sel = (alb_wvl_ext >= long_replace_start)
    alb_corr_fitted_ext_long[long_ext_sel] = interp_ori_spec_alb[long_ext_sel]
    alb_corr_fitted_ext[long_blend_sel | long_ext_sel] = np.clip(
        alb_corr_fitted_ext_long[long_blend_sel | long_ext_sel],
        0.0,
        1.0,
    )
    alb_corr_fitted_ext = _smooth_extension_tail(
        alb_wvl_ext,
        alb_corr_fitted_ext,
        start_wvl=long_replace_start,
    )
    
    
    # SW extension: 2nd-order polynomial fit to 450-600 nm gives a noise-free anchor
    # and is used to suppress detector-edge noise in the 350-450 nm native range.
    poly_win = (alb_wvl >= 450) & (alb_wvl <= 600) & np.isfinite(alb_corr_fitted)
    poly_fn = None
    poly_anchor_val = None
    if np.count_nonzero(poly_win) >= 3:
        coeffs = np.polyfit(alb_wvl[poly_win], alb_corr_fitted[poly_win], 2)
        poly_fn = np.poly1d(coeffs)
        poly_anchor_val = float(np.clip(poly_fn(short_wvl_start), 0.0, 1.0))

        # Step 2: blend polynomial into native 350-450 nm region.
        # Weight: 100% polynomial at 350 nm → 100% observed at 450 nm.
        blend_native_mask = (alb_wvl >= short_wvl_start) & (alb_wvl <= 450)
        if np.any(blend_native_mask):
            bwvl = alb_wvl[blend_native_mask]
            alpha_n = np.clip((bwvl - short_wvl_start) / (450.0 - short_wvl_start), 0.0, 1.0)
            blended_n = (
                (1.0 - alpha_n) * np.clip(poly_fn(bwvl), 0.0, 1.0)
                + alpha_n * alb_corr_fitted[blend_native_mask]
            )
            n_below = int(np.sum(alb_wvl_ext < alb_wvl[0]))
            alb_corr_fitted_ext[n_below + np.where(blend_native_mask)[0]] = np.clip(blended_n, 0.0, 1.0)

    # Step 3: anchor SNICAR extension at short_wvl_start using polynomial value if available,
    # otherwise fall back to the raw single point (original behaviour).
    anchor_idx = np.argmin(np.abs(alb_wvl_ext - short_wvl_start))
    if poly_anchor_val is not None:
        offset = poly_anchor_val - interp_snicar_unscaled[anchor_idx]
    else:
        short_wvl_sel = (alb_wvl >= short_wvl_start) & (alb_wvl <= short_wvl_end)
        offset = alb_corr_fitted[short_wvl_sel][0] - interp_snicar_unscaled[anchor_idx]

    # Step 4: fill extension below short_wvl_start with SNICAR+offset, blended near the
    # boundary toward the polynomial extrapolation for slope continuity.
    short_ext_sel = alb_wvl_ext < short_wvl_start
    ext_wvl_below = alb_wvl_ext[short_ext_sel]
    snicar_below = np.clip(interp_snicar_unscaled[short_ext_sel] + offset, 0.0, 1.0)
    if poly_fn is not None:
        blend_ext_width = 50.0  # nm below short_wvl_start over which to blend
        alpha_ext = np.clip((short_wvl_start - ext_wvl_below) / blend_ext_width, 0.0, 1.0)
        poly_below = np.clip(poly_fn(ext_wvl_below), 0.0, 1.0)
        # alpha_ext=0 at boundary (pure polynomial for continuity) → 1 farther away (SNICAR shape)
        alb_corr_fitted_ext[short_ext_sel] = (1.0 - alpha_ext) * poly_below + alpha_ext * snicar_below
    else:
        alb_corr_fitted_ext[short_ext_sel] = snicar_below
    
    # plt.close('all')
    # plt.figure(figsize=(8, 5))
    # plt.plot(alb_wvl, alb_corr_fitted, '-', color='gray', label='Corrected Albedo', linewidth=1)
    # plt.plot(short_wvl, short_wvl_alb, '-', color='k', label='Corrected Albedo', linewidth=3)
    # plt.plot(alb_wvl_ext[alb_wvl_ext < short_wvl_start], ext_array, '-', color='r', label='Best Fitted Albedo', linewidth=1.5)
    # plt.xlabel('Wavelength (nm)')
    # plt.ylabel('Albedo')
    # plt.legend()
    # plt.title(f'Short Wavelength Side Fitting')
    # plt.show()
    
    
    return alb_wvl_ext, alb_corr_fitted_ext


if __name__ == '__main__':


    pass
