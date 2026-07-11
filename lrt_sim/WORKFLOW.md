# lrt_sim Workflow Guide

End-to-end map of the ARCSIX SSFR surface-albedo / cloud-radiative-effect (CRE)
pipeline in `lrt_sim/`. Complements `REORG_PLAN.md` (file-move history) and
`cre/README.md` (CRE-specific detail).

> **Run everything from `lrt_sim/`** in the `er3t_env` conda environment.
> Paths resolve relative to `../data` on Mac and
> `/pl/active/vikas-arcsix/yuch8913/arcsix/data` on Linux/CURC
> (see `ssfr_atm_corr/settings.py`, the single source of truth for
> `_fdir_general_`, `_fdir_data_`, `_fdir_tmp_`, gas bands, and
> `ice_frac_time_offset`).

## Pipeline at a glance

```
in-situ prep (arcsix_gas_insitu.py, arcsix_cld_insitu.py, arcsix_marli_insitu.py)
        │  gas / cloud / MARLI profiles per date
        ▼
[1] ssfr_atm_corr.preprocess_runner
        │  data/flt_cld_obs_info/*_cld_obs_info_*_atm_corr.pkl   (per leg)
        ▼
[2] ssfr_atm_corr.runner  (iterative libRadtran atmospheric correction)
        │  data/lrt/<date>_<tag>_<sky>/ssfr_simu_flux_*_{iteration_N,final,final_extension}.csv
        │  data/sfc_alb/sfc_alb_*_{iter_N,final,final_extension}.dat
        ▼
[3] ssfr_atm_corr.processing_runner  (per-1-second products)
        │  data/sfc_alb_combined/sfc_alb_update_<date>_<tag>_time_*.pkl   (per case)
        ▼
[4] ssfr_atm_corr/combined.py  (season merge + collocation)
        │  data/sfc_alb_combined/sfc_alb_combined_spring_summer.pkl      ◄ the hand-off
        │  data/sfc_alb_combined/{alb_atm_corr,ssfr_atm_corrected}_combined_spring_summer.h5
        ▼
[5] cre/  (albedo files → SZA x CWP libRadtran sweep → CRE figures)
        │  data/sfc_alb_cre/*_cre_alb.dat, ext_alb_broadband.csv
        │  data/lrt/<date>_<tag>_<sky>_cre/ssfr_simu_flux_*_cre_{sw,lw}_sza_*.csv
        │  data/lrt/.../<date>_<tag>_cre_simulations_*.csv|.pkl, fig/<date>/*.png
        ▼
[6] analysis/ + ssfr_atm_corr/{analysis,spiral}.py  (figures off the combined product)
```

## Stage details

### 0. In-situ preparation (top-level scripts, run once per date)

| Script | Produces | Needed by |
|---|---|---|
| `arcsix_gas_insitu.py` | per-date trace-gas profiles (CO2/CH4/CO/O3) | atmosphere building in stages 2 and 5 |
| `arcsix_cld_insitu.py` | in-situ cloud LWC along track | cloud characterization |
| `arcsix_marli_insitu.py` | MARLI lidar cloud/water-vapor parameters | atmospheric profiles |

### 1. Preprocess — per-leg observation pickles

```bash
python -m ssfr_atm_corr.preprocess_runner <case_id ...>   # or --all / --clear-sky-all / --cloudy-all / --spiral-all
```

`ssfr_atm_corr/preprocess.py` loads HSK/SSFR/HSR1/MARLI/KT19 (paths via
`util.FlightConfig`), masks by pitch/roll and `ssfr_flags`, and writes one
pickle per leg to `data/flt_cld_obs_info/`. Legs failing QC (>30% all-NaN
wavelength channels) get a `{'skip': True}` sentinel that downstream stages
honor. QC figures go to `fig/<date>/` (`qc_plotting.py`).

### 2. Iterative atmospheric correction

```bash
python -m ssfr_atm_corr.runner <case_id ...> [--iterations 8] [--workers N] [--final-extension-rt]
```

Flow: `runner.py` → `case_catalog.run_catalog_case` → `workflow.flt_trk_atm_corr`
per iteration. Case definitions (date, time windows, clear/cloudy, manual cloud
microphysics, custom levels) live in `ssfr_atm_corr/case_catalog.py`
(`ALL_CASE_CATALOG` / `BAD_CASE_CATALOG`) — shared with the CRE package.

Per leg and iteration:
1. Build the atmosphere once (iter 0): MODIS-07 + nearest dropsonde + MARLI +
   in-situ gases via `util/arcsix_atm.prepare_atmospheric_profile` →
   `data/zpt/<date>/atm_profiles_*.dat` + `ch4_profiles_*.dat` (cached; reused
   on later iterations and by the CRE stage).
2. Run uvspec (`reptran coarse`, Bodhaine29 Rayleigh, SSFR solar flux, custom
   wavelength grid `wvl_grid_test.dat`).
3. Odell correction `alb_corr = alb_obs * (corr_dn / corr_up)` with gas-band
   masking (`helpers.gas_abs_masking`), SNICAR snow/ice fitting
   (`util/alb_fitting.snowice_alb_fitting`), and date-aware postfit cleanup
   (`postfit.py`; special handling for 20240603 / 20240807).
4. Closure check from iteration 2 (`DEFAULT_CLOSURE_THRESHOLDS` in
   `case_catalog.py`); on pass (or max additional iterations) the iteration is
   copied to `*_final.*`. **Final products require iter >= 2** (iter 1 is
   unfitted).
5. Optional `--final-extension-rt`: extended-grid (300-4000 nm) final RT pass.
   Requires the adjusted `sfc_alb_*_final_extension.dat` from stage 3 first —
   so the usual order is: runner → processing_runner → runner --final-extension-rt.

### 3. Processing — per-1-second albedo products

```bash
python -m ssfr_atm_corr.processing_runner <case_id ...> [--force-row-extension] [--no-plots]
```

`processing.py` reads the stage-2 CSV/`.dat` products and rebuilds per-second
albedo: native grid (352-1996 nm, `alb_final_all_1s`) and extended grid
(250-4050 nm @ 1 nm, `alb_final_ext_all_1s`). The extension scales the segment
final-extension template per row (or per-row SNICAR extension with
`--force-row-extension`), patches the native-trust range, and blends the
1996→2196 nm H2O-7 gap with a cosine taper (history in
`ssfr_atm_corr/ALBEDO_EXTENSION_NOTES.md`). Output: one
`sfc_alb_update_<date>_<tag>_time_<t0>_<t1>.pkl` per case, plus the adjusted
`sfc_alb_*_final_extension.dat` used by stage 2's extension RT pass.

### 4. Combine — season-level product

```bash
python ssfr_atm_corr/combined.py [--force] [--no-plots] [--no-collocation-plots]
```

Globs all `sfc_alb_update_*.pkl`, splits spring/summer at 2024-06-30, builds
`SeasonData`, collocates camera ice fraction (`analysis/ssfr_ice_frac_combined.py`
must have produced `data/cam_icefrac_rad/ice_frac_all.pkl` first), AMSR2 ice
concentration, ECICE/NSIDC ice age, ERA5 forecast albedo, and computes AART
grain sizes. Writes:

- `sfc_alb_combined_spring_summer.pkl` — the full combined dict consumed by
  the CRE package and all analysis scripts;
- `alb_atm_corr_combined_spring_summer.h5` and
  `ssfr_atm_corrected_combined_spring_summer.h5` — public HDF5 products
  (neutral "atm_corrected" naming).

### 5. CRE (see `cre/README.md` for full detail)

```bash
# Stage 5a: albedo .dat files from the declarative catalog
python -m cre.ext_alb_cases            # dry run
python -m cre.ext_alb_cases --apply    # write .dat (+ backup) + broadband CSV

# Stage 5b: libRadtran SZA x CWP sweep (SW/LW) for one cloud case
python -m cre.cre_runner --case-id case_004 --mode both [--manual-alb-sweep]
python -m cre.cre_runner --case-id case_019 --mode both --atm-file atm_profiles_... --manual-alb <file.dat>

# Stage 5c: post-process + figures
python cre/plot_case_004.py    # thin drivers around cre_plot.plot_cre_case
```

Key facts:
- Cloud microphysics / levels / time windows come from the shared
  `ssfr_atm_corr.case_catalog`; surface albedo comes from the combined product
  (per-leg pickle fallback for cases missing from it, e.g. `bad_case_003`).
- `cre_cases.MANUAL_ALB_SWEEP` is the curated 13-albedo sweep (broadband
  0.29-0.80). The SLURM scripts index into it so bash never drifts from Python.
- Runs are resumable: existing per-(albedo, SZA) CSVs and uvspec outputs are
  skipped unless `--overwrite-lrt`.
- `cre_plot.cre_sim_plot` builds SW/LW/net CRE vs LWP, the critical-LWP zero
  crossing, and the cos(SZA) x broadband-albedo critical-LWP contour (with the
  Shupe & Intrieri 2003 LWP=30 overlay).

### 6. Analysis figures

| Script | Input | Output |
|---|---|---|
| `ssfr_atm_corr/analysis.py` | combined pkl | per-case albedo/track figures under `fig/sfc_alb_corr_analysis/` |
| `analysis/ssfr_era5_alb_plot.py` | combined pkl + `data/era5/forecast_albedo_0_daily-mean.nc` | ERA5 collocation maps/scatter, `fig/ice_age/` |
| `analysis/ssfr_ice_frac_alb_analysis.py` | combined pkl + `ice_frac_all.pkl` | ice-fraction vs albedo regressions, grain size, `fig/sfc_alb_corr_analysis/<product>/` |
| `analysis/ssfr_R1_analysis.py`, `R1_*.py` | R1 HDF5 | SSFR QC figures |
| `ssfr_atm_corr/spiral.py` | per-leg pickles + `sfc_alb/*.dat` | per-leg/spiral diagnostics (interactive; uncomment blocks) |

## Cluster (CURC) notes

- `curc_shell_alpine_high_mem_cre_runner.sh` — SLURM array (one task per
  `MANUAL_ALB_SWEEP` entry) on the `amem` partition; sizes the uvspec worker
  pool by RAM (`MEM_PER_RUN_GB`, ~64 GB/run observed), and uses an atomic
  `mkdir` lock so exactly one array task prebuilds the shared atmospheric
  profile while the others wait. Edit `ATM_FILE` when switching cases.
- `curc_shell_blanca_cre_runner.sh` — preemptable/requeue variant for Blanca
  (tight ~2 GB/core memory; capped workers). One albedo index per submission.
- `cre/test_cre_curc.py` — single-SZA single-LWP end-to-end smoke test.
- Stage 2 memory scales ~linearly with `--workers` (parallel sub-legs).

## Directory layout (current, post-reorg)

```
lrt_sim/
  ssfr_atm_corr/     active atmospheric-correction package (stages 1-4)
  cre/               CRE package (stage 5)
  analysis/          ERA5 / ice-fraction / R1 / SI figure scripts (stage 6)
  util/              shared library: ICT readers, FlightConfig, SNICAR albedo
                     fitting (alb_fitting.py), MODIS+dropsonde atmosphere
                     builder (arcsix_atm.py)
  legacy/            pre-reorg backups (do not edit; reference only)
  data/, tmp/, fig/  runtime symlink-ish work dirs (inputs under ../data)
  *.dat / *.pkl      solar spectra (Kurucz, CU composite), wavelength grids,
                     SNICAR model libraries, cached intermediates
  arcsix_*.py, golden_lrt_*.py, map_legs*.py, ...
                     not-yet-reorganized top-level scripts; the arcsix_atm_corr /
                     toa_lrt_check / golden_lrt monoliths are superseded by the
                     packages, the in-situ prep + LW-test + map scripts are live
```

Divergences from `REORG_PLAN.md`: Phase-3 domains landed in a single
`analysis/` folder (not `era5_albedo/`, `ice_fraction/`, `r1_analysis/`,
`si_plots/`), and the CRE package is `cre/` (not `ssfr_cre/`) with modules
`cre_sim/cre_plot/cre_runner/cre_cases/ext_alb_cases` instead of
`workflow/plots/runner/helpers`.

## Maintenance: duplication / drift consolidation proposal

Several helpers exist in multiple copies with slightly different behavior, and
`from util import *` star-imports make silent shadowing easy. Verified
inventory (active code only; `legacy/` and the superseded top-level monoliths
excluded):

| Symbol | Canonical home | Duplicate copies | Notes |
|---|---|---|---|
| `fit_1d_poly` | (deleted from `helpers.py`) | `cre/cre_sim.py`, `ssfr_atm_corr/spiral.py` | both copies unused — safe to delete |
| `gas_abs_masking` | `ssfr_atm_corr/helpers.py` | `util/alb_fitting.py` (cached, star-exported), `analysis/ssfr_R1_analysis.py`, `spiral.py` | all live; behaviors differ (altitude-dependence defaults, fill strategy) |
| `ssfr_flags` | `ssfr_atm_corr/helpers.py` | `cre/cre_sim.py`, `analysis/ssfr_R1_analysis.py`, `spiral.py` | identical enum, copy-pasted |
| `write_2col_file` | `ssfr_atm_corr/helpers.py` (atomic) | `cre/cre_sim.py` (non-atomic; imported by `ext_alb_cases`), `spiral.py` | helpers version is strictly better (atomic replace) |
| gas-band constants | `ssfr_atm_corr/settings.py` | `cre/cre_sim.py` module top | **values differ** (h2o_1 672-706 vs 650-706; h2o_2 end 746 vs 760) but the cre_sim copy is unused |
| `solar_interpolation_func` | `util/util.py` | `cre/cre_sim.py` (unused), `spiral.py` | |
| `ssfr_time_series_plot` | `ssfr_atm_corr/qc_plotting.py` | `cre/cre_sim.py` (unused), `analysis/ssfr_R1_analysis.py` | |
| `make_default_config` | — | `ssfr_atm_corr/runner.py`, `processing.py`, `preprocess.py`, `cre/cre_sim.py`, `cre/cre_plot.py` | 5 near-identical copies |
| `find_catalog_case` | — | `cre/cre_sim.py`, `cre/cre_plot.py` | identical |
| `exp_decay` | — | `ssfr_atm_corr/workflow.py`, `cre/cre_sim.py` | both unused |

Proposed phases (no code changed yet — this is the plan of record):

- **Phase A — delete verified-dead copies (no behavior change).**
  In `cre/cre_sim.py`: `fit_1d_poly`, `ssfr_flags`, `ssfr_time_series_plot`,
  the gas-band constant block, `solar_interpolation_func`, `exp_decay`.
  In `spiral.py`: `fit_1d_poly`. In `workflow.py`: `exp_decay`.
- **Phase B — single-source the live duplicates.**
  `cre_sim`/`ext_alb_cases` import `write_2col_file` from
  `ssfr_atm_corr.helpers` (gains atomic writes); `cre_plot` imports
  `make_default_config`/`find_catalog_case` from `cre_sim`;
  `ssfr_R1_analysis` imports `ssfr_flags` from helpers. Do **not** blindly
  swap `gas_abs_masking` implementations — the variants differ; instead
  rename `util/alb_fitting.py`'s to a private `_gas_abs_masking` and drop it
  from the star export so shadowing becomes impossible, deferring true
  unification until the variants are reconciled.
- **Phase C — import hygiene (larger diff).** Replace `from util import *`
  in package modules (`workflow.py`, `cre_sim.py`, `combined.py`, ...) with
  explicit names; tighten `util/__init__.py` exports.
- **Known latent bug (flagged, not fixed):** `arcsix_lw_test.py` and
  `arcsix_lw_rad_test.py` call `fit_1d_poly` without defining or importing
  it — that code path raises `NameError` today.
