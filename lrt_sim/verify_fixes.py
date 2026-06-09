"""Diagnostic script to verify proposed fixes for Problems 1 and 2.

Verification 1 (SW native 350-450 nm):
  - Apply proposed polynomial blend to alb_final_all_1s row-by-row.
  - Compare mean native spectrum before/after in 300-600 nm.
  - Also run alb_extention on both the original and corrected native rows
    to check for double-blend in the extended array.

Verification 2 (LW trough at 2000 nm):
  - Re-run alb_extention on mean row with instrumented cap block.
  - Compare current vs. proposed ceiling approach in 1800-2500 nm.
"""

import pickle
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/Users/yuch8913/programming/arcsix_sfc/lrt_sim')
sys.path.insert(0, '/Users/yuch8913/programming/arcsix_sfc/lrt_sim/util')

from alb_fitting import alb_extention

PKL = (
    '/Users/yuch8913/programming/arcsix_sfc/data/'
    'sfc_alb_combined_smooth_450nm/'
    'sfc_alb_update_20240725_cloudy_atm_corr_time_15.094_15.300.pkl'
)
FIG_DIR = '/Users/yuch8913/programming/arcsix_sfc/lrt_sim'


def apply_sw_blend(native_wvl, rows):
    """Apply proposed Problem 1 fix: poly blend in 350-450 nm, row-by-row."""
    wvl = np.asarray(native_wvl, dtype=float)
    corrected = rows.copy()
    poly_mask = (wvl >= 450) & (wvl <= 600)
    blend_mask = (wvl >= 355) & (wvl <= 450)
    n_fixed = 0
    for i, row in enumerate(rows):
        valid = poly_mask & np.isfinite(row)
        if np.count_nonzero(valid) < 3:
            continue
        coeffs = np.polyfit(wvl[valid], row[valid], 2)
        poly_fn = np.poly1d(coeffs)
        bwvl = wvl[blend_mask]
        alpha = np.clip((bwvl - 355.0) / (450.0 - 355.0), 0.0, 1.0)
        blended = (1.0 - alpha) * np.clip(poly_fn(bwvl), 0.0, 1.0) + alpha * row[blend_mask]
        corrected[i][blend_mask] = np.clip(blended, 0.0, 1.0)
        n_fixed += 1
    return corrected, n_fixed


def finite_mean(arr):
    out = np.full(arr.shape[1], np.nan)
    for j in range(arr.shape[1]):
        col = arr[:, j]
        fin = col[np.isfinite(col)]
        if len(fin):
            out[j] = np.mean(fin)
    return out


def main():
    with open(PKL, 'rb') as f:
        d = pickle.load(f)

    native_wvl = np.array(d['native_wvl'])
    alb_native = np.array(d['alb_final_all_1s'])
    ext_wvl = np.array(d['extension_wvl_1s'])
    alb_ext = np.array(d['alb_final_ext_all_1s'])
    clear_sky = d.get('clear_sky', False)

    # ------------------------------------------------------------------ #
    # Verification 1 — SW native blend, and double-blend check            #
    # ------------------------------------------------------------------ #
    alb_native_fixed, n_fixed = apply_sw_blend(native_wvl, alb_native)
    print(f'V1: corrected {n_fixed}/{alb_native.shape[0]} rows in native SW region')

    mean_orig = finite_mean(alb_native)
    mean_fixed = finite_mean(alb_native_fixed)

    # Pick a representative finite row to run through alb_extention
    finite_rows = np.where(np.any(np.isfinite(alb_native), axis=1))[0]
    rep_idx = finite_rows[len(finite_rows) // 2]

    ext_wvl_from_orig,  ext_from_orig  = alb_extention(native_wvl, alb_native[rep_idx],       clear_sky=clear_sky)
    ext_wvl_from_fixed, ext_from_fixed = alb_extention(native_wvl, alb_native_fixed[rep_idx], clear_sky=clear_sky)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Verification 1 — SW native correction (case_023)', fontsize=13)

    # Panel A: mean native spectrum 300-650 nm
    ax = axes[0]
    sw = (native_wvl >= 300) & (native_wvl <= 650)
    ax.plot(native_wvl[sw], mean_orig[sw],  color='steelblue', lw=1.5, label='original')
    ax.plot(native_wvl[sw], mean_fixed[sw], color='tomato',    lw=1.5, label='after blend fix', ls='--')
    ax.axvspan(350, 450, alpha=0.1, color='orange', label='blend zone 350-450 nm')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Albedo')
    ax.set_title('Mean native albedo (300-650 nm)')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)

    # Panel B: single representative row, native 300-650 nm
    ax = axes[1]
    ax.plot(native_wvl[sw], alb_native[rep_idx][sw],       color='steelblue', lw=1.5, label='original row')
    ax.plot(native_wvl[sw], alb_native_fixed[rep_idx][sw], color='tomato',    lw=1.5, label='fixed row', ls='--')
    ax.axvspan(350, 450, alpha=0.1, color='orange', label='blend zone')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_title(f'Single row {rep_idx} native (300-650 nm)')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)

    # Panel C: extended array from original vs corrected native (300-650 nm) — double-blend check
    ax = axes[2]
    sw_ext_o = (ext_wvl_from_orig  >= 300) & (ext_wvl_from_orig  <= 650)
    sw_ext_f = (ext_wvl_from_fixed >= 300) & (ext_wvl_from_fixed <= 650)
    ax.plot(ext_wvl_from_orig[sw_ext_o],  ext_from_orig[sw_ext_o],  color='steelblue', lw=1.5, label='ext from orig native')
    ax.plot(ext_wvl_from_fixed[sw_ext_f], ext_from_fixed[sw_ext_f], color='tomato',    lw=1.5, label='ext from fixed native', ls='--')
    ax.axvspan(350, 450, alpha=0.1, color='orange', label='blend zone')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_title('Extended array (double-blend check, 300-650 nm)')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    out1 = f'{FIG_DIR}/verify_v1_sw_blend.png'
    fig.savefig(out1, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'V1 figure saved: {out1}')

    # ------------------------------------------------------------------ #
    # Verification 2 — LW cap: current vs. proposed                       #
    # ------------------------------------------------------------------ #
    # Use the mean spectrum as a representative row for the extended output
    mean_full_orig = finite_mean(alb_ext)

    # Re-run alb_extention with proposed cap on the representative row
    # We instrument alb_extention by monkey-patching the cap logic here.
    # Instead, we directly call alb_extention on the original native row
    # (current behaviour is already in ext_from_orig), then re-run with
    # the proposed fix applied post-hoc on the same SNICAR-scaled array.

    # To properly simulate the proposed fix we need the internal arrays.
    # We do this by temporarily adding a hook via a wrapper that exposes them.
    # Simplest approach: copy the relevant logic inline.

    from alb_fitting import _find_best_fit_cached
    from scipy.ndimage import uniform_filter1d

    def alb_extention_proposed(alb_wvl, alb_corr_fitted, clear_sky=False):
        """alb_extention with proposed Problem 2 cap replacing lines 789-799."""
        ext_wvl_start = 250
        ext_wvl_end = 4050
        ext_wvl_arr2 = np.arange(ext_wvl_start, ext_wvl_end + 1, 1)
        ext_alb = np.ones_like(ext_wvl_arr2) * alb_corr_fitted[-1]
        alb_wvl_ext = np.concatenate((
            ext_wvl_arr2[ext_wvl_arr2 < alb_wvl[0]],
            alb_wvl,
            ext_wvl_arr2[ext_wvl_arr2 > alb_wvl[-1]]
        ))
        alb_corr_fitted_ext = np.concatenate((
            ext_alb[ext_wvl_arr2 < alb_wvl[0]],
            alb_corr_fitted,
            ext_alb[ext_wvl_arr2 > alb_wvl[-1]]
        ))

        short_wvl_start, short_wvl_end = 355, 550
        long_wvl_start, long_wvl_end = 1450, 1740
        long_anchor_wvl = 1495.0
        long_blend_start = 1900.0
        long_scale_wvl = 2000.0
        long_replace_start = 2000.0

        long_wvl_sel = (
            (alb_wvl >= long_wvl_start) & (alb_wvl <= long_wvl_end)
            & np.isfinite(alb_corr_fitted)
        )
        if np.count_nonzero(long_wvl_sel) < 2:
            long_wvl_sel = (
                (alb_wvl >= 1500) & (alb_wvl <= 2000) & np.isfinite(alb_corr_fitted)
            )
        long_wvl = alb_wvl[long_wvl_sel]
        long_wvl_alb = alb_corr_fitted[long_wvl_sel]

        best_fit_key, best_fit_spectrum, min_rmse, ori_spec_wvl, ori_spec_alb = _find_best_fit_cached(
            clear_sky=clear_sky, obs_wvl=long_wvl, obs_albedo=long_wvl_alb
        )

        interp_ori_spec_alb = np.interp(alb_wvl_ext, ori_spec_wvl, ori_spec_alb)
        interp_snicar_unscaled = interp_ori_spec_alb.copy()

        finite_alb = np.isfinite(alb_corr_fitted)
        if np.count_nonzero(finite_alb) >= 2:
            obs_anchor = np.interp(long_anchor_wvl, alb_wvl[finite_alb], alb_corr_fitted[finite_alb])
        elif np.count_nonzero(finite_alb) == 1:
            obs_anchor = alb_corr_fitted[finite_alb][0]
        else:
            obs_anchor = np.nan
        model_scale_anchor = np.interp(long_scale_wvl, ori_spec_wvl, ori_spec_alb)
        model_anchor = np.interp(long_anchor_wvl, ori_spec_wvl, ori_spec_alb)
        if np.isfinite(obs_anchor) and np.isfinite(model_scale_anchor) and model_scale_anchor > 1e-6:
            interp_ori_spec_alb = interp_ori_spec_alb * (obs_anchor / model_scale_anchor)
        elif np.isfinite(obs_anchor) and np.isfinite(model_anchor):
            interp_ori_spec_alb = interp_ori_spec_alb + (obs_anchor - model_anchor)

        # --- PROPOSED cap (replaces lines 789-799) ---
        snicar_1495_mask = (alb_wvl_ext >= 1400) & (alb_wvl_ext <= 1550)
        snicar_post2000_mask = alb_wvl_ext >= long_replace_start
        blend_and_replace_mask = alb_wvl_ext >= long_blend_start

        if np.any(snicar_1495_mask) and np.any(snicar_post2000_mask):
            model_min_1495  = float(np.nanmin(interp_snicar_unscaled[snicar_1495_mask]))
            model_lm_post2000 = float(np.nanmax(interp_snicar_unscaled[snicar_post2000_mask]))
            if (
                model_min_1495 > 1e-6
                and np.isfinite(obs_anchor)
                and np.isfinite(model_lm_post2000)
            ):
                ceiling = obs_anchor * model_lm_post2000 / model_min_1495
                lm_actual = float(np.nanmax(interp_ori_spec_alb[blend_and_replace_mask]))
                if lm_actual > ceiling > 1e-6:
                    interp_ori_spec_alb[blend_and_replace_mask] *= ceiling / lm_actual
        # --- end proposed cap ---

        alb_corr_fitted_ext_long = np.copy(alb_corr_fitted_ext)
        long_blend_sel = (alb_wvl_ext > long_blend_start) & (alb_wvl_ext < long_replace_start)
        if np.any(long_blend_sel):
            alpha = ((alb_wvl_ext[long_blend_sel] - long_blend_start)
                     / (long_replace_start - long_blend_start))
            alpha = np.clip(alpha, 0.0, 1.0)
            alb_corr_fitted_ext_long[long_blend_sel] = (
                (1.0 - alpha) * alb_corr_fitted_ext[long_blend_sel]
                + alpha * interp_ori_spec_alb[long_blend_sel]
            )
        long_ext_sel = alb_wvl_ext >= long_replace_start
        alb_corr_fitted_ext_long[long_ext_sel] = interp_ori_spec_alb[long_ext_sel]
        alb_corr_fitted_ext[long_blend_sel | long_ext_sel] = np.clip(
            alb_corr_fitted_ext_long[long_blend_sel | long_ext_sel], 0.0, 1.0
        )

        # SW extension (unchanged from original)
        poly_win = (alb_wvl >= 450) & (alb_wvl <= 600) & np.isfinite(alb_corr_fitted)
        poly_fn = None
        poly_anchor_val = None
        if np.count_nonzero(poly_win) >= 3:
            coeffs = np.polyfit(alb_wvl[poly_win], alb_corr_fitted[poly_win], 2)
            poly_fn = np.poly1d(coeffs)
            poly_anchor_val = float(np.clip(poly_fn(short_wvl_start), 0.0, 1.0))
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

        anchor_idx = np.argmin(np.abs(alb_wvl_ext - short_wvl_start))
        if poly_anchor_val is not None:
            offset = poly_anchor_val - interp_snicar_unscaled[anchor_idx]
        else:
            short_wvl_sel = (alb_wvl >= short_wvl_start) & (alb_wvl <= short_wvl_end)
            offset = alb_corr_fitted[short_wvl_sel][0] - interp_snicar_unscaled[anchor_idx]

        short_ext_sel = alb_wvl_ext < short_wvl_start
        ext_wvl_below = alb_wvl_ext[short_ext_sel]
        snicar_below = np.clip(interp_snicar_unscaled[short_ext_sel] + offset, 0.0, 1.0)
        if poly_fn is not None:
            blend_ext_width = 50.0
            alpha_ext = np.clip((short_wvl_start - ext_wvl_below) / blend_ext_width, 0.0, 1.0)
            poly_below = np.clip(poly_fn(ext_wvl_below), 0.0, 1.0)
            alb_corr_fitted_ext[short_ext_sel] = (1.0 - alpha_ext) * poly_below + alpha_ext * snicar_below
        else:
            alb_corr_fitted_ext[short_ext_sel] = snicar_below

        return alb_wvl_ext, alb_corr_fitted_ext

    # Run both versions on the representative row
    ext_wvl_lw,  ext_orig_lw     = alb_extention(         native_wvl, alb_native[rep_idx], clear_sky=clear_sky)
    ext_wvl_lw2, ext_proposed_lw = alb_extention_proposed(native_wvl, alb_native[rep_idx], clear_sky=clear_sky)
    # Both should return the same wavelength grid
    ewvl = ext_wvl_lw

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Verification 2 — LW cap fix (case_023)', fontsize=13)

    # Panel A: full LW region 1700-2500 nm
    ax = axes[0]
    lw = (ewvl >= 1700) & (ewvl <= 2500)
    ax.plot(ewvl[lw], ext_orig_lw[lw],     color='steelblue', lw=1.5, label='current cap (lm_1750)')
    ax.plot(ewvl[lw], ext_proposed_lw[lw], color='tomato',    lw=1.5, label='proposed cap (SNICAR ratio)', ls='--')
    ax.axvline(1900, color='gray', lw=0.8, ls=':',  label='blend_start 1900')
    ax.axvline(2000, color='gray', lw=0.8, ls='--', label='replace_start 2000')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Albedo')
    ax.set_title('Extended albedo 1700-2500 nm')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)

    # Panel B: zoom on 1850-2150 nm — the discontinuity zone
    ax = axes[1]
    zoom = (ewvl >= 1850) & (ewvl <= 2150)
    ax.plot(ewvl[zoom], ext_orig_lw[zoom],     color='steelblue', lw=1.5, label='current')
    ax.plot(ewvl[zoom], ext_proposed_lw[zoom], color='tomato',    lw=1.5, label='proposed', ls='--')
    ax.axvline(1900, color='gray', lw=0.8, ls=':')
    ax.axvline(2000, color='gray', lw=0.8, ls='--')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_title('Zoom 1850-2150 nm (discontinuity check)')
    ax.legend(fontsize=8)

    # Panel C: diff
    ax = axes[2]
    lw2m = (ewvl >= 1700) & (ewvl <= 2500)
    diff = ext_proposed_lw[lw2m] - ext_orig_lw[lw2m]
    ax.plot(ewvl[lw2m], diff, color='purple', lw=1.5)
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.axvline(1900, color='gray', lw=0.8, ls=':')
    ax.axvline(2000, color='gray', lw=0.8, ls='--')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Δ albedo (proposed − current)')
    ax.set_title('Difference (proposed − current), 1700-2500 nm')

    plt.tight_layout()
    out2 = f'{FIG_DIR}/verify_v2_lw_cap.png'
    fig.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'V2 figure saved: {out2}')

    # Print key scalar diagnostics for V2
    mask_post2000 = ewvl >= 2000
    mask_1750_obs = (native_wvl >= 1650) & (native_wvl <= 1900) & np.isfinite(alb_native[rep_idx])
    lm_1750_obs = float(np.nanmax(alb_native[rep_idx][mask_1750_obs])) if np.any(mask_1750_obs) else np.nan
    print(f'V2 scalars for row {rep_idx}:')
    print(f'  obs lm_1750 (current ceiling proxy): {lm_1750_obs:.4f}')
    print(f'  orig  max post-2000 nm in ext:      {np.nanmax(ext_orig_lw[mask_post2000]):.4f}')
    print(f'  fixed max post-2000 nm in ext:      {np.nanmax(ext_proposed_lw[mask_post2000]):.4f}')
    idx_2000 = np.argmin(np.abs(ewvl - 2000.0))
    disc_orig  = float(ext_orig_lw[idx_2000]     - ext_orig_lw[idx_2000 - 1])
    disc_fixed = float(ext_proposed_lw[idx_2000] - ext_proposed_lw[idx_2000 - 1])
    print(f'  step at 2000 nm (current):  {disc_orig:+.5f}')
    print(f'  step at 2000 nm (proposed): {disc_fixed:+.5f}')


if __name__ == '__main__':
    main()
