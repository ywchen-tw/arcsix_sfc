# Albedo Extension: Design, Issues, and Fixes

## Overview

The atmospheric-correction pipeline produces two per-second albedo products:

| Array | Shape | Grid | Description |
|---|---|---|---|
| `alb_final_all_1s` | `(N, 380)` | native (352–1996.25 nm) | SNICAR-fitted surface albedo on the SSFR native wavelength grid |
| `alb_final_ext_all_1s` | `(N, 2537)` | ext_wvl (250–4050 nm, 1 nm) | Extended albedo covering the full SW spectrum |

`alb_iter2_ext_all` is a direct alias for `alb_final_ext_all_1s` (same object).

---

## Extension Function: `extend_final_albedo_1s`

**Location:** `lrt_sim/ssfr_atm_corr/processing.py`

Takes the native 1-second albedo array (`final_1s`) and extends it to the full `ext_wvl` grid using one of two code paths.

### Template path (`use_template=True`, `force_row_extension=False`)

Used when a valid segment-level `leg_extension` array is available.

1. Compute a per-row amplitude ratio `ratio_ext` from the native data relative to the segment mean (`leg_native_final`), using wavelengths below `h2o_7_start` (1748 nm) as the anchor region.
2. Scale the segment template: `row_ext = leg_extension * ratio_ext`
3. Apply the **`native_trust_range` patch**: overwrite `row_ext` at all ext_wvl points within the native wavelength range ([native_min, native_max]) with values interpolated directly from the per-1s native row. This ensures `alb_final_ext_all_1s` matches `alb_final_all_1s` within the native grid.
4. Apply `cap_longwave_extension` and clip to [0, 1].

### Per-row path (`force_row_extension=True` or `use_template=False`)

Used when the template is unavailable, or forced via `--force-row-extension` CLI flag.

1. Call `alb_extention(native_wvl, row)` per 1-second row to produce an extended spectrum.
2. Interpolate onto the common `ext_wvl` grid.
3. Apply `cap_longwave_extension`.
4. Apply the **`native_trust_range` patch** (same as template path): overwrite the native-range ext values with interpolated native values.
5. Apply the **cosine-taper gap blend** (see below).
6. Append to output.

---

## `alb_extention` Function

**Location:** `lrt_sim/util/alb_fitting.py`

Extends a single native albedo spectrum to the full 250–4050 nm grid using a SNICAR best-fit model.

Key internal wavelength boundaries:

| Variable | Value | Role |
|---|---|---|
| `long_blend_start` | 1900 nm | Start of native→SNICAR blend region |
| `long_replace_start` | 2000 nm | Full SNICAR replacement begins |
| `h2o_7_start` | 1748 nm | Start of H2O-7 gas absorption band |
| `h2o_7_end` | 2050 nm | End of H2O-7 gas absorption band |

**Scaling logic:**
- The SNICAR spectrum is initially scaled so that `SNICAR[native_max] = native[-1]` (last native observation before 2000 nm).
- A ceiling/floor adjustment then maps the full ≥1900 nm SNICAR range to `[obs_right_edge, ceiling]`, where `ceiling` is derived from the 1650–1900 nm window ratio.
- **Important:** This ceiling/floor adjustment can cause `SNICAR[native_max]` to deviate from `native[-1]` after the adjustment, since the floor is mapped to the *minimum* of the SNICAR in the ≥1900 nm region, which may not coincide with `native_max`.

**Result:** For clear-sky cases where gas absorption causes strong signal suppression in the 1748–1996 nm range, `alb_extention` values at 1991–1996 nm can be significantly higher than the native measurements (e.g., 0.074 vs. 0.057), because the SNICAR model does not include gas absorption.

---

## H2O-7 Band Gap (1997–2196 nm)

The native wavelength grid ends at 1996.25 nm, which is inside the H2O-7 absorption band (1748–2050 nm). Beyond 1996.25 nm, no native observations are available. `alb_extention` fills this region with scaled SNICAR, but the SNICAR values here can be substantially higher than the native observations in the gas absorption band.

This creates a visual and physical discontinuity at the 1996.25 → 1997 nm boundary after the native_trust_range patch overwrites the native range.

---

## Issues and Fixes

### Issue 1: `alb_final_ext_all_1s` does not match `alb_final_all_1s` at 1991–1996 nm (clear-sky cases)

**Symptom:** For clear-sky dates (e.g., 20240815), `alb_final_ext_all_1s` at 1991–1996 nm was ~0.074–0.097 while `alb_final_all_1s` showed ~0.057–0.077. Cloudy dates were unaffected.

**Root cause:** CURC was running `processing_runner.py --force-row-extension`, setting `force_row_extension=True`. This forced `use_template = False` in `extend_final_albedo_1s`, jumping to the per-row `alb_extention` path. The original per-row path did **not** apply the `native_trust_range` patch (lines 604–611 in the template path were never reached). The `native_trust_range = native_range` fix only existed in the template path.

**Why clear sky is uniquely affected:** For clear sky, gas absorption in the H2O-7 band causes native observations at 1991–1996 nm to be substantially lower than the SNICAR prediction. For cloudy cases, the SNICAR–native discrepancy is much smaller, so the missing patch was not visible.

**Fix:** Apply the `native_trust_range` patch to the per-row path as well. After calling `alb_extention` and `cap_longwave_extension`, overwrite `row_ext` at all ext_wvl points in `[native_min, native_max]` with values interpolated from the per-1s native row:

```python
finite_row = np.isfinite(row)
if _row_ntr is not None and np.any(_row_ntr) and np.count_nonzero(finite_row) >= 2:
    row_ext[_row_ntr] = np.interp(
        ext_wvl[_row_ntr],
        native_wvl[finite_row],
        row[finite_row],
        left=np.nan,
        right=np.nan,
    )
```

---

### Issue 2: Jump at ~1997 nm after native patch

**Symptom:** After applying the native_trust_range patch to the per-row path, a visible step appeared at ~1997 nm in `alb_final_ext_all_1s`. Values transitioned from `native[-1]` (~0.057) at 1996.25 nm to `alb_extention[1997]` (~0.075) at 1997 nm.

**Root cause:** The ceiling/floor adjustment in `alb_extention` causes the SNICAR spectrum at 1997 nm (still in the blend region, alpha=0.97) to be elevated above the native endpoint. After the native patch lowers 1996.25 nm to match native, the step to the unpatched 1997 nm value is visible.

**Attempted fix (discarded):** A linear blend from `native_endpoint` at 1996.25 nm to `alb_extention[h2o_7_end]` (2050 nm) as fixed right anchor. This moved the discontinuity to 2050 nm instead of removing it.

**Root cause of moved discontinuity:** The linear blend ended at a fixed right anchor value (`alb_extention[2050]`), but the SNICAR spectrum slope at 2050 nm differed from the linear blend slope, creating a kink at the blend endpoint.

---

### Issue 3: Jump at 2050 nm after linear gap blend

**Symptom:** After the linear gap blend, a kink appeared at exactly 2050 nm where the blend ended and `alb_extention` values resumed unchanged.

**Root cause:** A linear blend to a fixed right anchor value creates a slope discontinuity at the endpoint. The blend slope (rising from native_endpoint toward `alb_extention[2050]`) does not match the local slope of the `alb_extention` spectrum at 2050 nm (which is in the smoothed SNICAR region, potentially declining or flat).

**Fix:** Replace the fixed-anchor linear blend with a **cosine (Hann) taper** that blends from `native_endpoint` *toward* the existing `alb_extention` values — not toward a fixed target value. The key property: at alpha=1 (the right end of the 200 nm blend window), the blend equals `row_ext[gap_region]` exactly, so the derivative at the blend boundary automatically matches `alb_extention`'s derivative. No kink.

```python
native_max = np.nanmax(native_wvl)
blend_window = 200.0
left_idx = np.argmin(np.abs(ext_wvl - native_max))
native_endpoint = row_ext[left_idx]
gap_region = (ext_wvl > native_max) & (ext_wvl <= native_max + blend_window)
if np.any(gap_region) and np.isfinite(native_endpoint):
    alpha = np.clip((ext_wvl[gap_region] - native_max) / blend_window, 0.0, 1.0)
    alpha = (1.0 - np.cos(np.pi * alpha)) / 2.0
    row_ext[gap_region] = (1.0 - alpha) * native_endpoint + alpha * row_ext[gap_region]
```

**Properties of the cosine taper:**
- At `native_max` (left end): alpha=0, zero derivative → smooth connection to native patch
- At `native_max + 200 nm` (right end): alpha=1, blend = `alb_extention` value, slope = `alb_extention` slope → no kink
- Between: smooth S-curve transition through the H2O-7 gap region

**Blend window extent:** 1996.25 → 2196.25 nm, which covers the full H2O-7 gap (1997–2050 nm) and transitions into stable SNICAR territory well past the band end.

---

## Summary of Current Per-Row Path Logic

```
alb_extention(native_wvl, row)
    → raw extended spectrum (250–4050 nm, 1 nm grid)

cap_longwave_extension(ext_wvl, row_ext, row)
    → cap values beyond native_max (only active when native_max < h2o_7_start)

native_trust_range patch
    → overwrite ext_wvl ∈ [native_min, native_max] with interpolated native values
    → ensures alb_final_ext_all_1s matches alb_final_all_1s within native grid

cosine-taper gap blend (native_max, native_max + 200 nm]
    → smoothly blend from native endpoint into alb_extention values
    → eliminates step at the native grid boundary
```
