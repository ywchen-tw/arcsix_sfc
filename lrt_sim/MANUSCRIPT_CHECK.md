# Manuscript vs. data consistency check — 2026-07-11

Cross-check of `GRL-arcsix-albedo_draft_test_0711.docx` (+ SI) against the
current data products:

- combined albedo product `data/sfc_alb_combined/sfc_alb_combined_spring_summer.pkl`
  (regenerated 2026-06-12)
- CRE sweeps `data/lrt/*_cre/` for case_004 / case_014 / case_019
  (regenerated 2026-07-07/08)
- ice-fraction fits `data/sfc_alb_ice_frac/`, CRE albedos `data/sfc_alb_cre/`

## 1. Headline finding: §4.3 / Fig 4 numbers come from the old December run

Every critical-LWP number in §4.3 reproduces exactly in the OLD simulation of
the *other* June 3 cloud (`bad_case_003`, 13:37–13:45 CER-13.0 cloud, CSVs in
`data/lrt/20240603_cloudy_atm_corr_1_sat_cloud_ori/`, dated 2025-12-28), not in
the new `case_004` sweep that the Fig 4 caption describes:

| Quantity                   | Manuscript | New case_004 data                                  |
|----------------------------|-----------|-----------------------------------------------------|
| Observed SZA               | 61.46°    | **61.72°** (case mean; 61.52° for 14:43–14:45)      |
| Critical LWP, SSFR albedo  | 119.6     | **129.5 g m⁻²** (at albedo 0.758)                   |
| Critical LWP, ERA5 albedo  | 69.0      | **33.3 g m⁻²** (interpolated at ERA5 mean 0.651; the plotted figure value) |
| Measured cloud LWP         | 77.8      | **113.65 g m⁻²** (case_004 cloud; 77.82 = the 13:37–13:45 cloud) |

Old-run provenance: at SZA 61.93, albedo 0.704 → 119.6 and 0.655 → 69.0.

The qualitative story survives: observed LWP (113.65) still lies between the
ERA5 critical LWP (~30) and the SSFR critical LWP (129.5), so ERA5 still flips
the CRE sign (−38.4 W m⁻² vs **+2.2 W m⁻²** observed). Note the observed
warming is now marginal (+2.2 W m⁻²) — soften "confirm the cloud would
actually warm the surface" accordingly. The ERA5-marker shift (69 → ~30)
*strengthens* the misdiagnosis argument.

## 2. Values that check out against the new data

- **Fig 4a cloud** (case_004): CER 7.0 µm, CTH 1.91 km, CBH 0.50 km — matches
  the catalog and `fig/20240603/P3B_LWP_vs_Altitude_20240603_14.61_14.71.png`
  (cloud 0.504–1.906 km, LWP 113.65, COT 24.3).
- **Albedo 0.758 (0.30–4.0 µm)** for June 3 14:43–14:45: the
  `sfc_alb_20240603_14.716_14.749_0.34km_cre_alb.dat` file gives 0.7580.
- **Fig 2 (RF14, Aug 1)**: window 13:50:24–14:07:12 = the code's 13.84–14.12 h
  window; R² = 0.96 consistent (per-wavelength R² averages 0.958 over
  400–900 nm, max 0.966).
- **Fig S4.2 cloud**: CER 13.0, CTH 1.93, CBH 1.41, June 3 13:37–13:45 =
  `bad_case_003` exactly (LWP 77.82).
- **Fig S2.5**: LWP 24.1 / CER 8.4 matches the June 7 15.76–15.81 profile
  figure (LWP 24.07, CER 8.4) — but see §4 for the "baseline" claim.
- **"Critical LWP can exceed 100 g m⁻²"** holds: 115–281 g m⁻² for albedos
  0.751–0.797 at the case SZA.
- Per-case values from the regenerated figures (case-mean SZA now
  data-derived, commit 7c2b897):

  | Case              | Flight SZA | SSFR albedo → crit. LWP | ERA5 albedo → crit. LWP |
  |-------------------|-----------|--------------------------|--------------------------|
  | case_004 (Jun 3)  | 61.72°    | 0.758 → 129.5            | 0.651 → 30.3             |
  | case_014 (Jun 7)  | 61.47°    | 0.752 → 132.6            | 0.646 → 33.0             |
  | case_019 (Jun 13) | 60.40°    | 0.676 → 125.5            | 0.644 → 88.7             |

## 3. Values that no longer match the new data

- **Fig S3.1 (RF12 two-altitude closure)**: manuscript 0.669±0.019 (3.6 km) vs
  0.670±0.067 (0.1 km). Reproducing the exact selection (lat 83.9–85.0°,
  final 1-s broadband) from the new product gives **0.574±0.020 vs
  0.570±0.063**. Stds and altitudes (3.65/0.11 km) match; the means dropped
  ~0.10 — figure/text predate the June 12 product regeneration. Regenerate.
- **Saturated albedo range (Fig S4.1 / §4.2)**: manuscript 0.54–0.81.
  Solar-slit-weighted `alb_sif1` from the per-case fits
  (`data/sfc_alb_ice_frac/`), checked for both products (2026-07-13):
  * **native** (352–1996 nm): **0.548 (Aug 1) – 0.804 (Jun 5)** — matches the
    manuscript's 0.54–0.81 within rounding / the 5th–95th quantile bounds, so
    Fig S4.1 and the §4.2 text are evidently based on the native broadband.
  * **extended** (300–4000 nm): **0.534 (Aug 1) – 0.786 (Jun 5)** — the dark
    2–4 µm tail lowers every case by ~0.015–0.06.
  Not a stale-data issue, but §4.2 should state which product the range refers
  to, since the CRE/ERA5 sections use the extended broadband convention.
- **"Spring broadband albedo >0.7"**: new daily means (alt < 1.6 km) are
  0.728, 0.764, 0.800, 0.801, 0.764, **0.691 (Jun 11), 0.685 (Jun 13)** —
  consider "≈0.7 or above" / "mostly >0.7".

## 4. Internal inconsistencies (independent of data version)

- **Flight numbers around the ERA5 comparison**: the leg is June 3 = RF04
  (the manuscript itself calls the June 3 cloud "RF04"), but §4.3 says
  "during the RF07" / "a representative case on RF07" and the S4.3 caption
  says "during RF05". All should be RF04.
- **"cloud properties from SF07"** (§4.3, re Fig S4.2): the S4.2 caption's
  cloud is June 3 13:37–13:45 → RF04, not SF07.
- **S2.5's claim** that LWP 24.1/CER 8.4 "were used as the baseline for
  initializing the radiative transfer simulations in the main text": no
  current simulation uses that cloud. Main-text baseline = CER 7.0 /
  LWP 113.65 (case_004); case_014 uses 26.96/6.7.
- **Shupe & Intrieri year**: main text 2004 (correct: J. Climate 17, 616–628);
  S4.2 caption and the code's figure legend say 2003
  (`cre/cre_plot.py` "LWP=30 in Shupe and Intrieri (2003)").
- **FWHM**: main text "9.0 and 10.6 nm" vs SI S1.2.1 "9 nm to 12 nm".
- **SI cross-references scrambled**: S2.2 cites "Section S3.1/S3.3" (should be
  S2.1/S2.3); Section S3's text cites "Fig. S4.1/S3.2" for the captions'
  S3.1/S3.2; SSFR section mixes S1.2.x and S2.2.x numbering
  ("S2.2.3 Interspectrometer", "Figure S2.9/S2.10" for S1.9/S1.10).
- Conflict-of-interest section still contains the journal template text.

## 5. ERA5 albedo values (§4.3 deep-dive)

- ERA5 `fal` collocated to the Fig 4 leg: **mean 0.651** (0.648–0.658;
  effectively one static grid-box value → supports S4.3's "static,
  homogeneous" remark). 14:43–14:45 window: 0.648.
- "ERA5 systematically underestimates": confirmed — ERA5 < SSFR for 100% of
  leg seconds, and campaign-wide 86–100% (spring) / 59–100% (summer) of
  seconds per day; mean daily bias +0.04 to +0.24.
- **Weighting caveat**: the same leg has three "SSFR broadband" values —
  CRE `.dat` TOA-solar-slit weighted 0.748 (0.758 for the 2-min window);
  combined-product extended broadband (surface-flux weighted, cloudy sky)
  0.856; native 352–1996 nm broadband 0.764. The S4.3 caption should state
  which weighting is plotted (see the `fal` definition below: the
  TOA-weighted value is the direct `fal` comparison).
- Campaign means (alt ≤ 1.6 km, no spirals), TOA / flux-weighted vs ERA5:
  spring 0.717 / 0.751 vs 0.637; summer 0.537 / 0.570 vs 0.412.
  Cloudy legs only: spring 0.744 / 0.822 vs 0.647; summer 0.622 / 0.685 vs
  0.480 — the weighting gap roughly doubles under cloud, and ERA5's
  underestimate grows to ~0.18–0.21 under the actual-sky convention.
- New SI figures (commit d46c55f): `fig/SI/sfc_alb_ssfr_vs_era5_2panel`,
  `..._2panel_cloudy`, `..._5panel` — both weightings, ghost means, cloudy
  variant.

### What ERA5 `fal` is, and whether the comparison is sound (checked 2026-07-12)

The manuscript compares against ERA5 "forecast albedo" (`fal`, param 243) from
the reanalysis-era5-single-levels CDS dataset. Per the ECMWF radiation
documentation (Hogan 2015) and ERA5 docs:

- `fal` is a **diagnostic broadband albedo**: the model's **diffuse**
  UV-visible and near-IR surface albedos averaged with a **fixed
  top-of-atmosphere solar spectrum** — not weighted by the actual sky's
  downward flux. ECMWF notes it "differs somewhat from the true broadband
  all-sky albedo" for exactly these reasons.
- Consequently the **TOA-solar-weighted SSFR broadband is the
  apples-to-apples comparison with `fal`**; the ERA5 counterpart of the
  actual-sky flux-weighted SSFR broadband is the model's true all-sky albedo
  **α = 1 − SSR/SSRD** (both fluxes in the same single-levels dataset;
  undefined at night, and time-of-day dependent, so use hours near flight
  time rather than daily statistics).
- Over sea ice, `fal` follows the **Ebert & Curry (1993) monthly
  climatology** (dry snow Sep–May, melting snow June, bare ice Jul–Aug;
  mid-month values linearly interpolated), blended with open water by
  sea-ice concentration. It cannot respond to the actual surface state — so
  the SSFR−`fal` difference quantifies the bias of that climatology, which
  is precisely §4.3's argument. The comparison approach is well-precedented
  (Pohl et al. 2020 evaluated MERIS sea-ice albedo against `fal`).
- Manuscript wording to tighten: "the ERA5 albedo is from 0.2 to 4 µm" —
  more precisely, `fal` is a fixed-solar-spectrum weighted average of the
  model's UV-visible and near-IR diffuse albedos.

Sources:
- Hogan (2015), *Radiation Quantities in the ECMWF model and MARS*:
  <https://www.ecmwf.int/sites/default/files/elibrary/2015/18490-radiation-quantities-ecmwf-model-and-mars.pdf>
- ERA5 data documentation:
  <https://confluence.ecmwf.int/spaces/CKB/pages/76414402/ERA5+data+documentation>
- Pohl et al. (2020), *The Cryosphere*: <https://tc.copernicus.org/articles/14/165/2020/>

## 6. Follow-ups

- [ ] Update §4.3 / Fig 4 numbers to the new case_004 run (61.72°, 129.5,
      ~30, 113.65) — or re-run `bad_case_003` against the new albedo product
      if the 77.8 g m⁻² cloud should stay.
- [ ] Fix RF04/RF05/RF07/SF07 flight-number references.
- [ ] Regenerate Fig S3.1 (RF12) and Fig S4.1 from the June 12 product.
- [ ] Reconcile S2.5 "baseline" sentence with the actual case clouds.
- [ ] Shupe & Intrieri 2003 → 2004 in the S4.2 caption and the cre_plot
      figure legends.
- [ ] Fix SI section/figure cross-references and the FWHM statement.
- [ ] case_004 CRE sweep is missing albedo 0.676
      (`sfc_alb_20240613_14.109_14.140_0.11km_cre_alb.dat`, MANUAL_ALB_SWEEP
      index 6) — run `cre_runner` for it to complete the 13-albedo contour.
- [ ] S4.3 caption: state the SSFR weighting plotted; pair `fal` with the
      TOA-weighted broadband. Optionally download hourly `ssr`+`ssrd` for the
      flight days and add ERA5's true all-sky albedo (1 − SSR/SSRD) as the
      counterpart for the flux-weighted panels.
