# Manuscript vs. data consistency check

Cross-check of the GRL draft + SI against the current data products:

- combined albedo product `data/sfc_alb_combined/sfc_alb_combined_spring_summer.pkl`
  (regenerated 2026-06-12)
- CRE sweeps `data/lrt/*_cre/` for case_004 / case_014 / case_019
  (regenerated 2026-07-07/08)
- ice-fraction fits `data/sfc_alb_ice_frac/`, CRE albedos `data/sfc_alb_cre/`

History:
- 2026-07-11: first check against `GRL-arcsix-albedo_draft_test_0711.docx` (+ SI).
- 2026-07-13: re-check against `GRL-arcsix-albedo_draft_test_0713.docx` (+ SI).
  Most 0711 findings were fixed; §1 records what was resolved, §2 what remains.
- 2026-07-13 (later): re-check against `..._0713-2.docx` (+ SI). Nearly all
  remaining items fixed (see §1); §2 now holds only the short leftover list.

## 1. Resolved in the 0713 revision (verified against data)

- **§4.3 / Fig 4 updated from the old December `bad_case_003` run to case_004**:
  RF04, cloud window 14:36:36–14:42:36 UTC (14.61–14.71 h), CER 7.0 µm,
  CTH 1.91 km, CBH 0.50 km; measured LWP 113.7 g m⁻²; SSFR critical LWP 129.5;
  ERA5 critical LWP ~33 (figure value 33.3, interpolated at `fal` 0.651);
  SZA 61.5° for the 14:43–14:45 window. All match the new sweep. Both
  weighting conventions are now stated (0.758 TOA-weighted / 0.859
  simulated-surface-flux weighted, 0.30–4.0 µm) — both verified.
- **ERA5 `fal` handling adopted** (see §4 below for the background): §4.3 now
  explains the fixed-TOA-spectrum diffuse weighting, cites Hogan (2015), and
  pairs `fal` with the TOA-weighted SSFR broadband. The Data section cites
  "ERA5 post-processed daily statistics", which matches the file the code
  reads (`data/era5/forecast_albedo_0_daily-mean.nc`).
- **Fig S4.2 replaced** with the case_014 cloud: June 7 (RF07)
  15:14:24–15:17:24 (15.24–15.29 h), CER 6.7 µm, CTH 0.43 km, CBH 0.15 km,
  LWP 26.96 g m⁻² — verified against the case catalog and
  `fig/20240607/P3B_LWP_vs_Altitude_20240607_15.24_15.29.png` (cloud
  0.154–0.426 km). The "yields results similar to Shupe & Intrieri (2004)"
  claim is supported by the regenerated contour (the LWP=30 overlay tracks
  the simulated 25–30 g m⁻² contours). The old RF05/RF07/SF07 flight-number
  confusion around §4.3/S4.2/S4.3 is resolved (all RF04/RF07 now).
- **Fig S3.1 (RF12 two-altitude closure) values updated** to the June-12
  product: 0.574±0.020 (3.6 km) vs 0.570±0.063 (0.1 km).
- **FWHM** consistent: main text "about 9 and 12 nm", SI "9 nm to 12 nm".
- **Shupe & Intrieri year** fixed in the S4.2 caption (2004).
- **S4.3 caption** matches the new 5-panel SI figure (a = RF04 leg both
  weightings; b/c = cloudy legs TOA/flux; d/e = all legs TOA/flux) and states
  the weighting per panel.
- **Fig S4.1 caption** now states the native convention ("broadband albedo in
  the SSFR wavelength range").
- Abstract now says spring albedo "mostly exceeding" ERA5; §4.3 and the
  conclusions use ">0.68" (consistent with daily means 0.685–0.801).
- SI figure numbering S1.9–S1.11 now internally consistent; per-case values
  (case_004/014/019 table in §3 below) all reproduce.

## 2. 0713 open issues → status after the 0713-2 revision

Resolved in the **0713-2** revision (checked 2026-07-13):

- §4.2 saturated-albedo range corrected to **0.55 to 0.80** — matches the
  native per-day SIF=1 regression values (0.55 Aug 1 – 0.80 Jun 5).
- Abstract now ">0.68" (consistent with §4.3/conclusions and the June 11/13
  daily means 0.691/0.685).
- "observations confirm the cloud would **marginally** warm the surface" —
  softened as suggested (+2.2 W m⁻²).
- Fig 4 caption largely fixed: green star / red triangle in (a) now described
  and mapped to panel (c); stray "\" removed.
- Figure-side fixes (2026-07-13, `cre/cre_plot.py`): Shupe legend year
  2003 → 2004 in all figures (case_004/014/019 regenerated); Fig 4c
  ARCSIX/ERA5 markers moved to the observation-window SZA parsed from
  `obs_alb_file` (61.52° for 14:43–14:45, matching the quoted 61.5°). The
  panel-(a) curves and the 129.5/33.3 critical LWPs remain at the case-mean
  sweep SZA 61.72° (no 61.52° run in the grid) — ≲2 g m⁻² read-off
  difference.
- SI: reference list added; S2.4–S2.5 captions updated to the June 7
  15:14:24–15:17:24 cloud (LWP ~27, CER 6.7 — the S4.2 simulation cloud);
  cross-refs S1.2.7 → S1.2.2/S1.2.3 and S2.3 → "Section S2.2, step 6" fixed;
  captions now say RF12/RF05; S4.2 caption cross-references Fig. S2.5; the
  rewritten S4.3 caption adopted.
- Typos fixed: NSIDC, "cannot" (key point 2), "pixel-to-wavelength".

Still open after 0713-2:

- **§4.3 sentence**: "The **orange triangle** indicates the critical LWP on
  June 3rd from 14:43 to 14:45 … albedo of 0.758" — the 0.758/SSFR critical
  LWP is the **green star** in panel (a) (the red triangle is ERA5's) and,
  as of 2026-07-13, the **green star** in panel (c) too.
- **Fig 4 caption, marker mapping**: panel-(c) markers now reuse the
  panel-(a) colors (green star = SSFR, red triangle = ERA5; black edges;
  figures regenerated 2026-07-13), so the "They correspond to the triangle
  and star symbols in (c)" sentence — whose order was reversed anyway — can
  simply become "the same symbols/colors mark them in (c)". The **red
  circle** in (a) (ERA5 net CRE at the observed LWP, −38.4 W m⁻²) is still
  not mentioned.
- **References**: "(Copernicus Climate Change Service, Climate Data Store,
  2024)" still missing from the main reference list; **Pilewskie et al.
  (2003)** is cited in SI S1.2 but missing from the new SI reference list.
- **S2.4 caption says "June 3rd"** — the profiling maneuver containing the
  15:14:24–15:17:24 window is **June 7 (RF07)**, per the S2.5/S4.2 captions.
- **S2.5 small slips**: LWP "26.99" should be **26.96** g m⁻² (the profile
  figure's integrated value); "the radiative transfer simulations presented
  in the main text" — this cloud initializes the **Fig. S4.2 (SI)**
  simulation, not the main-text Fig. 4 one (which uses the June 3 cloud,
  LWP 113.65 / CER 7.0).
- Conflict-of-interest section is still the journal template text.
- Typo: "forcast albedo" (§4.3).

## 3. Reference values (verified 2026-07-11/13)

- **Fig 4a cloud** (case_004): CER 7.0 µm, CTH 1.91 km, CBH 0.50 km, LWP
  113.65 g m⁻², COT 24.3
  (`fig/20240603/P3B_LWP_vs_Altitude_20240603_14.61_14.71.png`).
- **Albedo 0.758 (0.30–4.0 µm, TOA-weighted)** for June 3 14:43–14:45:
  `sfc_alb_20240603_14.716_14.749_0.34km_cre_alb.dat` gives 0.7580.
  Flux-weighted counterpart 0.859±0.006 (window) / 0.856±0.011 (leg);
  native (352–1996 nm) 0.764.
- **Fig 2 (RF14, Aug 1)**: window 13:50:24–14:07:12 = the code's
  13.84–14.12 h window; R² = 0.96 consistent (per-wavelength R² averages
  0.958 over 400–900 nm, max 0.966).
- **"Critical LWP can exceed 100 g m⁻²"** holds: 115–281 g m⁻² for albedos
  0.751–0.797 at the case SZA.
- Per-case values (case-mean SZA data-derived, commit 7c2b897):

  | Case              | Flight SZA | SSFR albedo → crit. LWP | ERA5 albedo → crit. LWP |
  |-------------------|-----------|--------------------------|--------------------------|
  | case_004 (Jun 3)  | 61.72°    | 0.758 → 129.5            | 0.651 → 33.3 (interp.)   |
  | case_014 (Jun 7)  | 61.47°    | 0.752 → 132.6            | 0.646 → 33.0             |
  | case_019 (Jun 13) | 60.40°    | 0.676 → 125.5            | 0.644 → 88.7             |

- Saturated albedo (SIF=1, solar-slit-weighted spectral fits in
  `data/sfc_alb_ice_frac/`): native 0.55–0.80 (per-day regression, Fig S4.1),
  extended 0.56–0.86; the dark 2–4 µm tail lowers most days, cloudy
  (diffuse) days sit higher in the extended product.
- Old-run provenance of the superseded 0711 numbers (61.46°, 119.6, 69.0,
  77.8): `bad_case_003` (June 3 13:37–13:45 CER-13.0 cloud, CSVs in
  `data/lrt/20240603_cloudy_atm_corr_1_sat_cloud_ori/`, dated 2025-12-28),
  at SZA 61.93: albedo 0.704 → 119.6 and 0.655 → 69.0.

## 4. ERA5 albedo values (§4.3 deep-dive)

- ERA5 `fal` collocated to the Fig 4 leg: **mean 0.651** (0.648–0.658;
  effectively one static grid-box value → supports S4.3's "static,
  homogeneous" remark). 14:43–14:45 window: 0.648.
- "ERA5 systematically underestimates": confirmed — ERA5 < SSFR for 100% of
  leg seconds, and campaign-wide 86–100% (spring) / 59–100% (summer) of
  seconds per day; mean daily bias +0.04 to +0.24.
- Campaign means (alt ≤ 1.6 km, no spirals), TOA / flux-weighted vs ERA5:
  spring 0.717 / 0.751 vs 0.637; summer 0.537 / 0.570 vs 0.412.
  Cloudy legs only: spring 0.744 / 0.822 vs 0.647; summer 0.622 / 0.685 vs
  0.480 — the weighting gap roughly doubles under cloud, and ERA5's
  underestimate grows to ~0.18–0.21 under the actual-sky convention.
- SI figures (commit d46c55f): `fig/SI/sfc_alb_ssfr_vs_era5_2panel`,
  `..._2panel_cloudy`, `..._5panel` — both weightings, ghost means, cloudy
  variant.

### What ERA5 `fal` is, and whether the comparison is sound (checked 2026-07-12)

The manuscript compares against ERA5 "forecast albedo" (`fal`, param 243),
daily-mean statistics from the CDS. Per the ECMWF radiation documentation
(Hogan 2015) and ERA5 docs:

- `fal` is a **diagnostic broadband albedo**: the model's **diffuse**
  UV-visible and near-IR surface albedos averaged with a **fixed
  top-of-atmosphere solar spectrum** — not weighted by the actual sky's
  downward flux. ECMWF notes it "differs somewhat from the true broadband
  all-sky albedo" for exactly these reasons.
- Consequently the **TOA-solar-weighted SSFR broadband is the
  apples-to-apples comparison with `fal`** (now adopted in §4.3); the ERA5
  counterpart of the actual-sky flux-weighted SSFR broadband is the model's
  true all-sky albedo **α = 1 − SSR/SSRD** (both fluxes in the same
  single-levels dataset; undefined at night, and time-of-day dependent, so
  use hours near flight time rather than daily statistics).
- Over sea ice, `fal` follows the **Ebert & Curry (1993) monthly
  climatology** (dry snow Sep–May, melting snow June, bare ice Jul–Aug;
  mid-month values linearly interpolated), blended with open water by
  sea-ice concentration. It cannot respond to the actual surface state — so
  the SSFR−`fal` difference quantifies the bias of that climatology, which
  is precisely §4.3's argument. The comparison approach is well-precedented
  (Pohl et al. 2020 evaluated MERIS sea-ice albedo against `fal`).

Sources:
- Hogan (2015), *Radiation Quantities in the ECMWF model and MARS*:
  <https://www.ecmwf.int/sites/default/files/elibrary/2015/18490-radiation-quantities-ecmwf-model-and-mars.pdf>
- ERA5 data documentation:
  <https://confluence.ecmwf.int/spaces/CKB/pages/76414402/ERA5+data+documentation>
- Pohl et al. (2020), *The Cryosphere*: <https://tc.copernicus.org/articles/14/165/2020/>

## 5. Follow-ups

- [x] §4.2 saturated-albedo range corrected to 0.55–0.80 (0713-2).
- [x] Abstract ">0.7" → ">0.68" (0713-2).
- [x] "confirm … warm" softened to "marginally warm" (0713-2).
- [x] Shupe & Intrieri 2003 → 2004 in the `cre_plot.py` figure legends;
      Fig 4, contour-only variants, and case_014/case_019 regenerated
      (2026-07-13).
- [x] Fig 4c markers moved to the observation-window SZA (61.52°)
      (2026-07-13).
- [x] SI reference list added; SI cross-refs, SF#→RF#, stray "\", most typos
      fixed (0713-2).
- [x] Fig 4c / contour-only markers recolored to match panel (a)
      (green star = SSFR, red triangle = ERA5) and regenerated (2026-07-13).
- [ ] §4.3 sentence: "orange triangle … 0.758" → "green star"; Fig 4 caption:
      replace the "correspond to the triangle and star" sentence with "the
      same symbols and colors mark them in (c)" and mention the ERA5 obs-LWP
      red circle.
- [ ] Add the ERA5 CDS reference to the main list; add Pilewskie et al.
      (2003) to the SI list.
- [ ] S2.4 caption "June 3rd" → June 7; S2.5 "26.99" → 26.96; point the
      "baseline" sentence at Fig. S4.2 rather than the main text.
- [ ] CoI template text; "forcast albedo" typo.
- [x] case_004 CRE sweep albedo 0.676
      (`sfc_alb_20240613_14.109_14.140_0.11km_cre_alb.dat`): CURC CSVs
      downloaded to `data/lrt/20240603_cloudy_atm_corr_2_sat_cloud/`, copied
      into the `..._sat_cloud_cre/` dir the plots read, and all case_004
      figures regenerated with the full 13-albedo contour (2026-07-13).
      Values unchanged: ERA5 0.651 still interpolates between 0.638 and
      0.672 → critical LWP 33.3; case_014/case_019 already had 0.676.
- [ ] Optional: download hourly `ssr`+`ssrd` for the flight days and add
      ERA5's true all-sky albedo (1 − SSR/SSRD) as the counterpart for the
      flux-weighted SI panels.
