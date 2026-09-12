# Albedo Refinement Fix Plan

## Problems

### Problem 1 — iter 1 (Odell-corrected) spikes at gas absorption band edges

**Location:** `workflow.py:749–763`

The Odell correction computes:
```
alb_corr = alb_obs × (corr_dn / corr_up)
  where corr_dn = F_dn_p3_sim / F_dn_obs
        corr_up = F_up_p3_sim / F_up_obs
```

Both `corr_dn` and `corr_up` are computed at **every** wavelength including gas absorption
bands, where `F_dn_obs ≈ 0`. The correction factor blows up there, producing spikes that
get clipped to 1.0 but corrupt the Odell-corrected albedo (iter 1). This is the root cause
of all downstream problems in the SNICAR fitting step.

---

### Problem 2 — iter 2 (SNICAR-fitted) wrong plateau at 1200–1500 nm

**Location:** `alb_fitting.py:247`, `alb_fitting.py:311–313`

`band_5_fit` is defined as 1185–1700 nm. This single band straddles **two separate** gas
absorption gaps: h2o_5 (1230–1286 nm) and h2o_6 (1290–1509 nm). With two masked regions
inside one fitting band, `bandfit_nan` spans nearly the entire band. The left/right anchor
points are at 1185 nm and 1700 nm respectively (albedo ~0.75 and ~0.2), and the SNICAR
rescaling fills the large gap with a flat plateau instead of the correct declining NIR shape.

---

### Problem 3 — iter 2 discontinuous jumps at gas-band edges

**Location:** `alb_fitting.py:321–393`

The current gap-fill anchoring uses a **mean of 5 observed points** for `xl_origin`/`xr_origin`
but a **single SNICAR point** for `xl_fit`/`xr_fit`. These refer to different wavelengths,
so the linear rescaling does not guarantee that `replace_array[0]` equals the observed value
at the exact left edge of the gap, or `replace_array[-1]` at the right edge. This produces
visible discontinuities at every gas-band boundary in the fitted iter 2 albedo.

---

## Proposed Changes

### Change 1 — `workflow.py:748–778`
**Mask `corr_dn` and `corr_up` before the Odell multiplication**

For the current conservative implementation, mask every configured gas band in both
`corr_dn` and `corr_up` at every altitude. Fill masked correction factors with 1.0
(identity) rather than letting them blow up. A reduced low-altitude `corr_up` mask remains
available behind `ALTITUDE_DEPENDENT_GAS_MASKING=True` for future evaluation.

```python
corr_up = Fup_p3_mean_interp / fup_mean
corr_dn = Fdn_p3_mean_interp / fdn_mean

# corr_dn: always mask — downwelling always traverses the full atmospheric column
corr_dn_masked = gas_abs_masking(alb_wvl, corr_dn, alt=999.0)
corr_dn_filled = np.where(np.isnan(corr_dn_masked), 1.0, corr_dn_masked)

# corr_up: full masking now; optional reduced low-altitude mask behind settings switch
corr_up_masked = gas_abs_masking(
    alb_wvl,
    corr_up,
    alt=alt_avg,
    altitude_dependent=ALTITUDE_DEPENDENT_GAS_MASKING,
)
corr_up_filled = np.where(np.isnan(corr_up_masked), 1.0, corr_up_masked)

alb_corr = alb_obs * (corr_dn_filled / corr_up_filled)
alb_corr[:4] = alb_corr[4]
alb_corr[alb_corr < 0.0] = 0.0
alb_corr[alb_corr > 1.0] = 1.0
# existing NaN-fill loop below retained as safety net
```

> **Optional (future):** Physics justification for the altitude threshold is documented
> at the bottom of this file.

---

### Change 2 — `alb_fitting.py:247–257` and `311–313`
**Split `band_5_fit` so each sub-band spans at most one gas absorption gap**

```python
# Replace:
alb_wvl_sep_5th_s, alb_wvl_sep_5th_e = 1185, 1700
band_5_fit = (alb_wvl >= alb_wvl_sep_5th_s) & (alb_wvl < alb_wvl_sep_5th_e)

# With:
band_5a_fit = (alb_wvl >= 1185) & (alb_wvl < 1290)  # bridges h2o_5 (1230–1286) only
band_5b_fit = (alb_wvl >= 1285) & (alb_wvl < 1520)  # bridges h2o_6 (1290–1509) only
band_5c_fit = (alb_wvl >= 1515) & (alb_wvl < 1700)  # clean window, no internal gap
```

Update the fitting loop accordingly:
```python
for bands_fit in [band_2_fit, band_3_fit, band_4_fit,
                  band_5a_fit, band_5b_fit, band_5c_fit, band_6_fit]:
```

---

### Change 3 — `alb_fitting.py:321–393`
**Replace mean-based anchoring with exact 2-point affine solve**

Use the exact single point immediately before the gap (`bandfit_nan_ind[0]-1`) and
immediately after (`bandfit_nan_ind[-1]+1`) as the anchors for both the observed albedo
and the SNICAR spectrum. Solve the linear system `a * snicar + b = obs` at both boundary
points, which guarantees continuity with zero jump at either edge.

```python
bandfit_nan = np.isnan(alb_corr_mask[bands_fit])
if bandfit_nan.sum() == 0:
    continue
bandfit_nan_ind = np.where(bandfit_nan)[0]
if bandfit_nan_ind[-1] == len(bandfit_nan) - 1:
    bandfit_nan_ind = bandfit_nan_ind[:-1]

band_len = int(np.sum(bands_fit))
has_right = bandfit_nan_ind[-1] + 1 < band_len

# Exact single boundary points
xl_obs    = alb_corr_fit[bands_fit][bandfit_nan_ind[0] - 1]
xl_snicar = best_fit_spectrum[bands_fit][bandfit_nan_ind[0] - 1]
xr_obs    = alb_corr_fit[bands_fit][bandfit_nan_ind[-1] + 1] if has_right else xl_obs
xr_snicar = best_fit_spectrum[bands_fit][bandfit_nan_ind[-1] + 1] if has_right else xl_snicar

# Solve: a * xl_snicar + b = xl_obs  AND  a * xr_snicar + b = xr_obs
denom = xr_snicar - xl_snicar
if abs(denom) > 1e-6:
    a = (xr_obs - xl_obs) / denom
    b = xl_obs - a * xl_snicar
    replace_array = a * best_fit_spectrum[bands_fit][bandfit_nan] + b
else:
    # SNICAR is flat in this gap — fall back to linear interpolation between observed edges
    wvl_gap = alb_wvl[bands_fit][bandfit_nan]
    wvl_l   = alb_wvl[bands_fit][bandfit_nan_ind[0] - 1]
    wvl_r   = alb_wvl[bands_fit][bandfit_nan_ind[-1] + 1] if has_right else wvl_l + 1.0
    replace_array = np.interp(wvl_gap, [wvl_l, wvl_r], [xl_obs, xr_obs])

alb_corr_fit_replace = copy.deepcopy(alb_corr_fit[bands_fit])
alb_corr_fit_replace[bandfit_nan] = replace_array
alb_corr_fit[bands_fit] = alb_corr_fit_replace
```

---

## Summary

| # | File | Lines | Fixes | Status |
|---|------|--------|-------|--------|
| 1 | `workflow.py` | 763–777 | iter 1 spikes at gas-band edges | ✅ Done (2026-06-02) |
| 2 | `alb_fitting.py` | 255–258, 313–315 | iter 2 plateau at 1200–1500 nm | ✅ Done (2026-06-02) |
| 3 | `alb_fitting.py` | 317–348 | iter 2 jumps at gas-band boundaries | ✅ Done (2026-06-02) |

---

## Optional — Physics justification for altitude-dependent `corr_up` masking

### Current implementation status

For now, all configured gas absorption bands are masked at every altitude. Both active
`gas_abs_masking` implementations default to:

```python
altitude_dependent=False
```

The reduced mask below 0.5 km is retained as an optional future experiment and can be
enabled for the workflow by setting:

```python
ALTITUDE_DEPENDENT_GAS_MASKING = True
```

or enabled for an individual helper call with:

```python
gas_abs_masking(wvl, alb, alt=alt, altitude_dependent=True)
```

O2-A remains masked in both full and reduced-mask modes.

### Geometry of the two flux measurements at altitude z

**Downwelling (zenith, F_dn) at any altitude z:**
- Light travels TOA → z, through the entire atmospheric column above z
- At 0.31 km: τ_down ≈ τ_total (nearly full column above aircraft)
- True at ALL altitudes: F_dn always sees the full column

**Upwelling (nadir, F_up) at altitude z:**
- Light is reflected from surface and travels through only the z km of atmosphere below the aircraft
- At 0.31 km: τ_up ≈ 0.04 × τ_O2_total and ~15% of τ_H2O_total
- F_up is barely affected by gas absorption at low altitude

### Why separating them is physically justified

At 0.31 km in a gas absorption band:
```
F_dn(z=0.31km) ≈ F_dn(surface)                              # thin layer, nearly same flux
F_up(z=0.31km) ≈ α_sfc × F_dn(surface) × exp(-τ_up/0.31km) ≈ α_sfc × F_dn(surface)
```

So `alb_obs = F_up/F_dn ≈ α_sfc` — the naive ratio is approximately the true surface
albedo even at gas band wavelengths at low altitude, because neither flux is uniquely
depleted relative to the other.

| Component | At 0.31 km in gas band | Problem? |
|-----------|------------------------|----------|
| `corr_dn = F_dn_sim / F_dn_obs` | Both ≈ 0 (full column above) | Yes — ratio unstable, sensitive to RTM inaccuracy |
| `corr_up = F_up_sim / F_up_obs` | Both ≈ α_sfc × F_dn_sfc (thin path up) | No — ratio well-conditioned, close to 1 |

The spike problem in iter 1 is caused entirely by `corr_dn` blowing up, not `corr_up`.
At low altitude `corr_up` is stable at gas bands because neither the simulation nor the
observation has significant upward gas attenuation.

The 0.5 km threshold is grounded in the water vapor scale height (~2.5 km):
`exp(-0.5/2.5) ≈ 18%` of column below — a reasonable cutoff where the upwelling path
starts introducing non-trivial absorption in the NIR H2O bands. For O2-A (scale height
~8 km) the threshold could be higher, but 0.5 km is conservative.

The altitude-dependent reduced mask remains useful as a future experiment, but the current
conservative default masks all configured gases for both correction factors and all
altitudes. Enable the reduced low-altitude behavior only with
`ALTITUDE_DEPENDENT_GAS_MASKING=True`.
