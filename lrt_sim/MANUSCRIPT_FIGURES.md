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
`data/sfc_alb_combined/sfc_alb_combined_spring_summer.pkl` (regenerated 2026-06-12)
and need no libRadtran run.

## Main text

| Figure | Content | Output file | Script |
|---|---|---|---|
| Fig 1a,b | Spring/summer broadband-albedo maps over AMSR2 SIC | `fig/sfc_alb_corr_lonlat/arcsix_broadband_albedo_vs_longitude_polar_projection_spring_summer_combined.png` | [`ssfr_atm_corr/combined.py`](ssfr_atm_corr/combined.py) (`fig_dir` set at ~line 1981, savefig ~line 2340) |
| Fig 1c | Daily mean spectral albedo per science flight (gas bands shaded/filled) | `fig/sfc_alb_corr_lonlat/arcsix_albedo_all_flights.png` (variants: `..._partial`, `..._clear_partial`, `..._cloudy_partial`, `..._myi`) | [`ssfr_atm_corr/combined.py`](ssfr_atm_corr/combined.py) |
| Fig 1d | Daily broadband albedo vs daily camera sea-ice fraction (error bars = std) | `fig/sfc_alb_corr_lonlat/arcsix_broadband_albedo_690_1190_vs_Sea_Ice_Fraction.png` (camera SIF; the `..._vs_NSIDC_Sea_Ice_Fraction.png` sibling uses NSIDC SIC instead) | [`ssfr_atm_corr/combined.py`](ssfr_atm_corr/combined.py) (~lines 2410–2530) |
| Fig 2 | RF14 (2024-08-01, 13.84–14.12 UTC h) albedo & SIC time series, linear fit + quantile shading, 4 camera images | `fig/sfc_alb_corr_analysis/{native,extended}/arcsix_albedo_0801_clear_broadband_icefraction_combined.{png,pdf}` | [`analysis/ssfr_ice_frac_alb_analysis.py`](analysis/ssfr_ice_frac_alb_analysis.py) |
| Fig 3 | Saturated (SIF=1) broadband albedo vs KT19 surface T (a) and multi-year-ice coverage (b) | `fig/sfc_alb_corr_analysis/{native,extended}/arcsix_albedo_broadband_vs_kt19_myi_ratio.{png,pdf}` | [`analysis/ssfr_ice_frac_alb_analysis.py`](analysis/ssfr_ice_frac_alb_analysis.py) |
| Fig 4 | CRE vs LWP (a), albedo spectra (b), critical-LWP contour (c) for case_004 | `fig/20240603/surface_net_cre_lwp_and_contour_20240603_cloudy_atm_corr_2_combined.{png,pdf}` | [`cre/cre_plot.py`](cre/cre_plot.py) via [`cre/plot_case_004.py`](cre/plot_case_004.py) |

## Supporting Information

| Figure | Content | Output file | Script |
|---|---|---|---|
| S1.1 | P-3 flight tracks, both campaigns | `arcsix_flight_paths_all.png` (written to cwd) | [`map_legs_all.py`](map_legs_all.py) |
| S1.2–S1.3 | Wavelength calibration / slit-function Gaussian fits | not in this repo — SSFR calibration workflow (external) | — |
| S1.4 | Primary response functions (nad/zen × Si/InGaAs) | `fig/SI/pri_response_ori.png`, `pri_response_ori_2.png` | [`analysis/ssfr_SI_plot.py`](analysis/ssfr_SI_plot.py) |
| S1.5–S1.9 | Field-lamp stability, transfer spectra, cosine response | not in this repo — SSFR calibration workflow (external) | — |
| S1.10 | ALP attitude-corrected direct irradiance | not in this repo (SSFR/ALP processing) | — |
| S1.11 | SSFR/TOA flux-ratio stability per flight | per-flight `fig/{date}/{date}_{case}_toa_dnflux_toa_ratio.png` — **verify**; related checks in `R1_flag_check.py`, `R1_ssfr_zen_nad_compare.py` | [`arcsix_toa_lrt_check.py`](arcsix_toa_lrt_check.py) |
| S2.1–S2.2 | Skew-T + WVMR composite-profile comparisons | not found in this repo — likely produced during atm-profile prep (external/notebook) | — |
| S2.3 | Trace-gas vertical profiles (H2O/CH4/CO2/O3) | `../data/zpt/{date}/{date}_gases_profiles.png` | [`arcsix_gas_insitu.py`](arcsix_gas_insitu.py) |
| S2.4–S2.5 | Cloud profiling maneuver time series; LWC/extinction profiles (S2.5 values LWP 24.07 g m⁻², CER 8.4 μm) | `fig/20240607/P3B_LWP_vs_Altitude_20240607_15.76_15.81.{png,pdf}` (other legs: `fig/{date}/P3B_LWP_vs_Altitude_{date}_{t0}_{t1}.png`) | [`arcsix_cld_insitu.py`](arcsix_cld_insitu.py) |
| S3.1 | RF12 (2024-07-29) two-altitude closure: tracks + mean spectra + broadband vs latitude | `fig/sfc_alb_corr_analysis/arcsix_albedo_0729_clear_1_summary.{png,pdf}` | [`ssfr_atm_corr/analysis.py`](ssfr_atm_corr/analysis.py) `combined_atm_corr()` |
| S3.2 | RF05 (2024-06-05) spiral descent version of S3.1 | `fig/sfc_alb_corr_analysis/arcsix_albedo_0605_clear_spiral_summary.{png,pdf}` | [`ssfr_atm_corr/analysis.py`](ssfr_atm_corr/analysis.py) |
| S4.1 | Saturated broadband albedo per day/case with 5th–95th quantile bounds, clear vs cloudy | `fig/sfc_alb_corr_analysis/{native,extended}/arcsix_albedo_broadband_ice_frac_fit_summary.{png,pdf}` | [`analysis/ssfr_ice_frac_alb_analysis.py`](analysis/ssfr_ice_frac_alb_analysis.py) |
| S4.2 | Critical-LWP contour for the 2024-06-03 13:37–13:45 cloud (CER 13.0 μm; `bad_case_003`) | `fig/20240603/surface_net_cre_lwp_and_contour_20240603_cloudy_atm_corr_1_*.png` — current version is from the **Dec 2025** run (`data/lrt/20240603_cloudy_atm_corr_1_sat_cloud_ori/`); re-run `cre` for `bad_case_003` to refresh | [`cre/cre_plot.py`](cre/cre_plot.py) |
| S4.3 | SSFR vs ERA5 broadband albedo: (a) case_004 leg, (b) all legs TOA-weighted, (c) all legs actual-sky-flux weighted | `fig/SI/sfc_alb_ssfr_vs_era5_2panel.{png,pdf}` | [`analysis/ssfr_era5_alb_si_fig.py`](analysis/ssfr_era5_alb_si_fig.py) |

## Regeneration cheat-sheet

| To refresh… | Run |
|---|---|
| Fig 4 (and per-case CRE diagnostics) | `python cre/plot_case_004.py` (needs er3t PYTHONPATH; reads cached per-SZA CSVs) |
| Figs 2, 3, S4.1 | `python analysis/ssfr_ice_frac_alb_analysis.py` (combined product + camera netCDFs) |
| S3.1/S3.2 | `python -c "from ssfr_atm_corr.analysis import combined_atm_corr; combined_atm_corr()"` (or run the module's `__main__`) |
| S4.3 | `python analysis/ssfr_era5_alb_si_fig.py` (combined product only; no er3t needed) |
| S2.4/S2.5 | `python arcsix_cld_insitu.py` (in-situ cloud ICT files) |
| Fig 1 panels | `ssfr_atm_corr/combined.py` full run (heavy: rebuilds/loads combined product + collocations) |

Notes:
- The current draft's §4.3 numbers (critical LWP 119.6/69.0, SZA 61.46°, LWP 77.8 g m⁻²)
  trace to the old `bad_case_003` run; the regenerated case_004 figure gives
  SZA 61.72°, critical LWP ≈ 130 (SSFR 0.758) vs ≈ 30 (ERA5 0.651), observed LWP 113.65 g m⁻².
- Rows marked **verify** were not fully confirmed against the draft; check the
  basename when the figure is next regenerated.
