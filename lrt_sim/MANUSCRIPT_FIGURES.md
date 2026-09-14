# Manuscript figure map

Where each figure of *"High Arctic Sea Ice Albedo Shifts Liquid Cloud Radiative Effect
Toward Surface Warming in Boreal Spring"* (GRL draft + SI) lives on disk and which
script regenerates it. Output paths are relative to `lrt_sim/` unless noted;
`fig/sfc_alb_corr_analysis/` has `native/` and `extended/` subfolders (same figure,
352–1996 nm vs 300–4000 nm albedo product).

All scripts below share the GRL style in [`plot_style.py`](plot_style.py)
(170 mm full width, Arial 8–10 pt, ≥0.25 pt lines, 300 dpi PNG + vector PDF,
Okabe–Ito / cividis colors).

## How to run

```bash
# some scripts import lrt_sim/util, which needs the main er3t checkout:
PYTHONPATH="/Users/yuch8913/programming/er3t/er3t:$PYTHONPATH" \
  /Users/yuch8913/miniforge3/envs/er3t_env/bin/python <script>
```

Scripts marked "combined product" read
`data/sfc_alb_combined/sfc_alb_combined_spring_summer.pkl` (regenerated 2026-09-13, from SSFR R1)
and need no libRadtran run.

## Main text

| Figure | Content | Output file | Script |
|---|---|---|---|
| Fig 1a,b | Spring/summer broadband-albedo maps over AMSR2 SIC. The **whole of Fig 1 (a–d)** is this one file — panels c and d are drawn inside it, so copy it directly; the standalone `arcsix_albedo_all_flights.png` / `..._690_1190_vs_*.png` below are separately styled working versions, not the manuscript panels | `fig/sfc_alb_corr_lonlat/arcsix_broadband_albedo_vs_longitude_polar_projection_spring_summer_combined.png` | [`ssfr_atm_corr/combined.py`](ssfr_atm_corr/combined.py) (`fig_dir` set at ~line 1981, savefig ~line 2340) |
| Fig 1c | Daily mean spectral albedo per science flight (gas bands shaded/filled) | `fig/sfc_alb_corr_lonlat/arcsix_albedo_all_flights.png` (variants: `..._partial`, `..._clear_partial`, `..._cloudy_partial`, `..._myi`) | [`ssfr_atm_corr/combined.py`](ssfr_atm_corr/combined.py) |
| Fig 1d | Daily broadband albedo (690–1190 nm) vs daily camera sea-ice fraction (error bars = 5th–95th percentile of each day's distribution, both axes; caption must not say "std") | `fig/sfc_alb_corr_lonlat/arcsix_broadband_albedo_690_1190_vs_Sea_Ice_Fraction.png` (camera SIF; the `..._vs_NSIDC_Sea_Ice_Fraction.png` sibling uses NSIDC SIC instead) | [`ssfr_atm_corr/combined.py`](ssfr_atm_corr/combined.py) (~lines 2410–2530) |
| Fig 2 | RF14 (2024-08-01, 13.84–14.12 UTC h) albedo & SIC time series, linear fit + quantile shading, 4 camera images. **Native** product (Fig 2b: R²=0.96, y=0.46x+0.11; the extended sibling reads 0.95) | `fig/sfc_alb_corr_analysis/native/arcsix_albedo_0801_clear_broadband_icefraction_combined.{png,pdf}` | [`analysis/ssfr_ice_frac_alb_analysis.py`](analysis/ssfr_ice_frac_alb_analysis.py) |
| Fig 3 | Ice-pack (SIF=1) broadband albedo vs over-ice KT-19 surface T (nadir HDRF ≥ 0.4; linear + flat-then-ramp fits, schematic CICE ramp) (a) and vs multi-year-ice coverage (b); high-SIF-mean legs ringed. **Native** product (T₀=−3.2 °C, hinge R²=0.75 vs linear 0.57, F-test p=0.008; the extended sibling gives T₀=−1.7 °C and backs the §4.2 CMP22 band-matched campaign comparison, 0.56–0.86) | `fig/sfc_alb_corr_analysis/native/arcsix_albedo_broadband_vs_kt19_myi_ratio_kt19fill.{png,pdf}` (the `_kt19fill` suffix: KT-19 medians gap-filled for the panel-b coloring; identical hinge fit to the plain `..._myi_ratio.png`, but that plain variant drops the panel-b colorbar and so does **not** match the caption) | [`analysis/ssfr_ice_frac_alb_analysis.py`](analysis/ssfr_ice_frac_alb_analysis.py) |
| Fig 4 | CRE vs LWP (a), albedo spectra (b), critical-LWP contour (c) for case_004 | `fig/20240603/surface_net_cre_lwp_and_contour_20240603_cloudy_atm_corr_2_combined.{png,pdf}` | [`cre/cre_plot.py`](cre/cre_plot.py) via [`cre/plot_case_004.py`](cre/plot_case_004.py) |

## Supporting Information

| Figure | Content | Output file | Script |
|---|---|---|---|
| S1 (new first SI figure; former S1–S20 shifted to S2–S21) | R0 vs R1 downwelling closure "before/after": spectra, TOA-normalized, sim residual (RF05 high-altitude clear leg) | `fig/SI/ssfr_r0_r1_before_after.{png,pdf}` → `GRL-manuscript/si_figures/figS01_r0_r1_closure.png` | [`analysis/ssfr_R0_R1_si_fig.py`](analysis/ssfr_R0_R1_si_fig.py) (needs `data/processed/ARCSIX-SSFR_P3B_R0/`) |
| S1.1 | P-3 flight tracks, both campaigns | `arcsix_flight_paths_all.png` (written to cwd) | [`map_legs_all.py`](map_legs_all.py) |
| S1.2–S1.3 | Wavelength calibration / slit-function Gaussian fits | not in this repo — SSFR calibration workflow (external) | — |
| S1.4 | Primary response functions (nad/zen × Si/InGaAs) | `fig/SI/pri_response_ori_2.png` → `figS05` (LaTeX `R^{pri(ori)}` labels, `ncol=4`); the `pri_response_ori.png` sibling is the older plain-text-label version and is **not** the one in the SI | [`analysis/ssfr_SI_plot.py`](analysis/ssfr_SI_plot.py) |
| S1.5–S1.9 | Field-lamp stability, transfer spectra, cosine response | not in this repo — SSFR calibration workflow (external) | — |
| S1.10 | ALP attitude-corrected direct irradiance | not in this repo (SSFR/ALP processing) | — |
| S1.11 | SSFR/TOA flux-ratio stability per flight | per-flight `fig/{date}/{date}_{case}_toa_dnflux_toa_ratio.png` — **verify**; related checks in `R1_flag_check.py`, `R1_ssfr_zen_nad_compare.py` | [`arcsix_toa_lrt_check.py`](arcsix_toa_lrt_check.py) |
| S2.1–S2.2 | Skew-T + WVMR composite-profile comparisons | not found in this repo — likely produced during atm-profile prep (external/notebook) | — |
| S2.3 | Trace-gas vertical profiles (H2O/CH4/CO2/O3) | `../data/zpt/{date}/{date}_gases_profiles.png` | [`arcsix_gas_insitu.py`](arcsix_gas_insitu.py) |
| S2.4–S2.5 | Cloud profiling maneuver time series (S2.4); LWC/extinction profiles (S2.5). RF07 15:14:24–15:17:24 UTC leg — SI quotes LWP ≈27 g m⁻², CER 6.7 μm | S2.4 `fig/20240607/P3B_insitu_4panel_20240607_15.24_15.29.{png,pdf}`; S2.5 `fig/20240607/P3B_LWP_vs_Altitude_20240607_15.24_15.29.{png,pdf}` (**not** the 15.76_15.81 leg; other legs: `fig/{date}/P3B_LWP_vs_Altitude_{date}_{t0}_{t1}.png`) | [`arcsix_cld_insitu.py`](arcsix_cld_insitu.py) |
| S3.1 | RF12 (2024-07-29) two-altitude closure: tracks + mean spectra + broadband vs latitude | `fig/sfc_alb_corr_analysis/arcsix_albedo_0729_clear_1_summary.{png,pdf}` | [`ssfr_atm_corr/analysis.py`](ssfr_atm_corr/analysis.py) `combined_atm_corr()` |
| S3.2 | RF05 (2024-06-05) spiral descent version of S3.1 | `fig/sfc_alb_corr_analysis/arcsix_albedo_0605_clear_spiral_summary.{png,pdf}` | [`ssfr_atm_corr/analysis.py`](ssfr_atm_corr/analysis.py) |
| S4.1 | Ice/snow end member (SIF=1 broadband albedo) per day/case with 5th–95th bounds, clear vs cloudy, ARCSIX legs only (`..._noref` variant). Sibling variants keep the published MOSAiC/N-ICE2015 end-member overlays: `..._summary`, and `..._summary_lat` whose legend also gives the latitude range each published end member came from (MOSAiC 78.6–89.1°N, N-ICE2015 80–83°N; ARCSIX legs span 78.8–85.7°N). Collocated ERA5 is deliberately **not** drawn — ERA5 `fal` is a grid-box mean blended with open water, not an ice end member (see S4.3 for the like-for-like comparison) | `fig/sfc_alb_corr_analysis/native/arcsix_albedo_broadband_ice_frac_fit_summary_noref.{png,pdf}` — **native** product, matching Fig 3 and the §4.2 range 0.56–0.81 (extended sibling and overlay variants `..._summary{,_lat}.{png,pdf}` in both product folders) | [`analysis/ssfr_ice_frac_alb_analysis.py`](analysis/ssfr_ice_frac_alb_analysis.py); literature values and latitudes from [`../data/SI_data/endmember_lit_values.csv`](../data/SI_data/endmember_lit_values.csv) |
| S4.2 | Critical-LWP contour for the RF07 (2024-06-07) 15:14:24–15:17:24 cloud (CER 6.7 μm, top 0.43 km, base 0.15 km) | `fig/20240607/surface_cre_vs_lwp_all_alb_20240607_cloudy_atm_corr_combined_contour_only_no_symbol.png` (Aug 2025 CRE run) | [`cre/cre_plot.py`](cre/cre_plot.py) |
| S4.3 | SSFR vs ERA5 broadband albedo, **5-panel**: (a) RF04 leg, (b,c) cloudy legs only, (d,e) all legs; TOA-solar weighting in (b,d) and actual-sky-flux in (c,e) | `fig/SI/sfc_alb_ssfr_vs_era5_5panel.{png,pdf}` (**not** the `_2panel` sibling) | [`analysis/ssfr_era5_alb_si_fig.py`](analysis/ssfr_era5_alb_si_fig.py) |

## Regeneration cheat-sheet

| To refresh… | Run |
|---|---|
| Fig 4 (and per-case CRE diagnostics) | `python cre/plot_case_004.py` (needs er3t PYTHONPATH; reads cached per-SZA CSVs) |
| Figs 2, 3, S4.1 | `python analysis/ssfr_ice_frac_alb_analysis.py` (combined product + camera netCDFs) |
| S3.1/S3.2 | `python -m ssfr_atm_corr.analysis` (runs `combined_atm_corr()`; cwd must be `lrt_sim/`). ⚠️ Three files define `combined_atm_corr()`; only `ssfr_atm_corr/analysis.py` is live. `ssfr_atm_corr_analysis.py` (top level, no slash) and `legacy/ssfr_atm_corr_analysis.py` read `data/sfc_alb_combined_smooth_450nm/`, which does not exist, and crash. `ssfr_atm_corr/combined.py` *builds* the combined product but does not plot these. The savefig calls use `save_grl()`, so grepping for `savefig` misses them. Regenerates 8 summary figures, not just these two. |
| S4.3 | `python analysis/ssfr_era5_alb_si_fig.py` (combined product only; no er3t needed) |
| S2.4/S2.5 | `python arcsix_cld_insitu.py` (in-situ cloud ICT files) |
| Fig 1 panels | `ssfr_atm_corr/combined.py` full run (heavy: rebuilds/loads combined product + collocations) |

Notes:
- **Manuscript copies (`GRL-manuscript/figures/`, `si_figures/`).** These are copies, not
  symlinks, so they go stale whenever a source is re-run. Synced 2026-09-14 from the
  2026-09-13 products: `fig02`←`native/arcsix_albedo_0801_clear_broadband_icefraction_combined.png`,
  `fig03`←`native/arcsix_albedo_broadband_vs_kt19_myi_ratio_kt19fill.png`,
  `figS01`←`SI/ssfr_r0_r1_before_after.png`, `figS04`←`SI/ssfr_lamp_color_temp_derivation.png`,
  `figS05`←`SI/pri_response_ori_2.png`, `figS19`←`SI/arcsix_vs_sheba_mosaic.png`,
  `figS20`←`native/arcsix_albedo_broadband_ice_frac_fit_summary_noref.png`,
  `figS21`←`SI/sfc_alb_ssfr_vs_era5_5panel.png`,
  `fig01`←`sfc_alb_corr_lonlat/arcsix_broadband_albedo_vs_longitude_polar_projection_spring_summer_combined.png`
  (all four panels in one file). Still outstanding: **figS17/figS18**, whose sources are from
  2026-06-12 and have never been re-run against the R1 combined product. `figS02, S03, S06, S07, S09–S14` come from
  the external SSFR calibration workflow and cannot be checked from this repo.
- **Ice/snow end member (SIF=1).** Normally the OLS extrapolation of albedo vs camera
  sea-ice fraction. Where a leg's SIF spread (p95−p5) is below 0.10 *and* the broadband
  fit has r² < 0.50, the extrapolated slope is a leverage artifact (2024-05-28 spans
  0.001 in SIF and fits a slope of −1.58 at 500 nm), so the end member falls back to the
  mean of the SIF > 0.98 points. One gate per leg, applied to every SIF=1 quantity
  (broadband, 1240/1700 nm, ratio-method grain size, spectral `alb_sif1`). 5 of 17 legs
  fall back; the values move by ≤0.001, so Figs 2/3 and S4.1 are unchanged in substance.
  Per-leg decisions and the numbers behind them: `../data/sfc_alb_ice_frac/{native,extended}/sif1_method_audit.csv`;
  the fit CSVs carry an `alb_sif1_method` column. Fallback legs are ringed in S4.1.
- **Ice/snow end-member comparison (§4.1–4.3).** `analysis/ssfr_ice_frac_alb_analysis.py`
  prints, per leg, the collocated ERA5 albedo and ERA5 − α(SIF=1). Extended product:
  spring end member 0.697–0.858, summer 0.571–0.769;
  ERA5 − α(SIF=1) = −0.139 spring, −0.257 summer, most negative −0.342.
  Native product: 0.677–0.806 / 0.563–0.705 and −0.108 / −0.235, most negative −0.350
  (2024-08-15). (2026-09-13 run.)
  Compare Di Biagio et al. (2021, §3.1.2): ERA5 up to −0.25 against their N-ICE2015
  point albedo and up to −0.1 against the SIC-weighted grid albedo, after their day 145
  (25 May) — every ARCSIX flight falls after that date. The aggregate-albedo counterpart
  (−0.062 spring, −0.114 summer; n = 18,005 / 21,517) comes from
  [`analysis/era5_ssfr_bias.py`](analysis/era5_ssfr_bias.py).
- **§4.3 / Fig 4 (case_004).** The figure is from the Aug 2025 CRE run and gives
  SZA 61.72°, critical LWP ≈ 130 (SSFR 0.742) vs ≈ 37 (ERA5 0.651), observed LWP 113.7 g m⁻².
  The draft still says "SZA 61.5°" — the figure and the sweep both use 61.72°.
  The albedo `.dat` inputs were rewritten 2026-09-13 and still give 0.742 / 0.651 / 0.732 /
  0.735 to three decimals, so the CRE numbers are very likely unchanged — but the sweep
  itself predates them and the per-SZA output CSVs (`data/lrt/*_cre/`) are **not on this
  machine**, so Fig 4 and S4.2 cannot be regenerated locally without re-running libRadtran.
- Rows marked **verify** were not fully confirmed against the draft; check the
  basename when the figure is next regenerated.
