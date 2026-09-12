# LRT Simulation Reorganization Plan

This note tracks the planned cleanup of `lrt_sim`. The goal is to separate active workflows, products, plotting, and legacy backups without changing scientific behavior during the move.

## Guiding Principles

- Move one domain at a time, starting with the active SSFR atmospheric-correction workflow.
- Keep original backup files available in `legacy/`.
- Use short module names inside domain folders instead of long repeated prefixes.
- Keep executable scripts named `runner.py`.
- Keep main scientific orchestration named `workflow.py`.
- Keep shared cross-domain utilities in `util/` or a future `common/` module.

## Target Structure

```text
lrt_sim/
  ssfr_atm_corr/
    __init__.py
    runner.py              # run selected atmospheric-correction catalog cases
    workflow.py            # flt_trk_atm_corr and final spectral simulation logic
    case_catalog.py        # CASE_CATALOG, SPIRAL_CASE_CATALOG, closure checks
    settings.py            # paths, mission/platform constants, gas bands
    setup.py               # wavelength grids, support files, dropsonde/cloud loading
    helpers.py             # small math, IO, flags, masking, fitting helpers
    qc_plotting.py         # SSFR time-series and QC plots
    preprocess.py          # cloud-observation input-product generation
    preprocess_runner.py   # run preprocessing for selected catalog cases
    processing.py          # final/final_extension atmospheric-correction products
    processing_runner.py   # run post-processing for selected catalog cases
    spiral.py              # spiral-case plotting workflow

  ssfr_cre/
    __init__.py
    runner.py              # CRE batch runs, if needed
    workflow.py            # CRE radiative-transfer workflow
    plots.py               # CRE plotting
    helpers.py             # CRE-local helpers

  era5_albedo/
    __init__.py
    runner.py
    workflow.py
    plots.py
    helpers.py

  ice_fraction/
    __init__.py
    runner.py
    workflow.py
    plots.py
    helpers.py

  r1_analysis/
    __init__.py
    runner.py
    workflow.py
    plots.py
    helpers.py

  si_plots/
    __init__.py
    runner.py
    plots.py
    helpers.py

  legacy/
    ssfr_atm_corr_ori.py
    ssfr_atm_corr_plot.py
    ssfr_atm_corr_processing.py
    ssfr_atm_corr_processing_ori.py
    ssfr_atm_corr_product_helpers.py
    ssfr_data_collect_ori.py
    # other historical scripts preserved for reference
```

## Current To Target Mapping

### Phase 1: SSFR Atmospheric Correction

Status: completed. Active atmospheric-correction modules now live under `lrt_sim/ssfr_atm_corr/`, and `ssfr_atm_corr_ori.py` is preserved under `lrt_sim/legacy/`.

```text
ssfr_atm_corr_case_runner.py     -> ssfr_atm_corr/runner.py
ssfr_atm_corr_workflow.py        -> ssfr_atm_corr/workflow.py
ssfr_atm_corr_case_catalog.py    -> ssfr_atm_corr/case_catalog.py
ssfr_atm_corr_settings.py        -> ssfr_atm_corr/settings.py
ssfr_atm_corr_setup.py           -> ssfr_atm_corr/setup.py
ssfr_atm_corr_helpers.py         -> ssfr_atm_corr/helpers.py
ssfr_atm_corr_qc_plotting.py     -> ssfr_atm_corr/qc_plotting.py
ssfr_data_collect.py             -> ssfr_atm_corr/preprocess.py
ssfr_atm_corr_processing.py      -> ssfr_atm_corr/processing.py
ssfr_atm_corr_plot.py            -> ssfr_atm_corr/spiral.py
ssfr_atm_corr_ori.py             -> legacy/ssfr_atm_corr_ori.py
ssfr_data_collect_ori.py         -> legacy/ssfr_data_collect_ori.py
ssfr_atm_corr_plot.py            -> legacy/ssfr_atm_corr_plot.py backup
ssfr_atm_corr_processing.py      -> legacy/ssfr_atm_corr_processing.py backup
ssfr_atm_corr_product_helpers.py -> legacy/ssfr_atm_corr_product_helpers.py
```

Expected run command after Phase 1:

```bash
python3 -m lrt_sim.ssfr_atm_corr.runner
```

### Phase 2: Products And CRE

Status: in progress. The active atmospheric-correction product builder now lives
inside `ssfr_atm_corr/` because it depends directly on the catalog, final native
SSFR-grid files, and final extended-grid files. The old product helper module is
kept only with the legacy processing backup.

```text
ssfr_atm_corr_processing.py          -> ssfr_atm_corr/processing.py
ssfr_atm_corr_product_helpers.py     -> legacy/ssfr_atm_corr_product_helpers.py
ssfr_atm_corr_processing_ori.py      -> legacy/ssfr_atm_corr_processing_ori.py
ssfr_cre.py                         -> ssfr_cre/workflow.py
ssfr_cre_plot.py                    -> ssfr_cre/plots.py
```

### Phase 3: Other Analysis Domains

```text
ssfr_era5_alb_plot.py               -> era5_albedo/plots.py
ssfr_ice_frac_alb_analysis.py       -> ice_fraction/workflow.py
ssfr_ice_frac_combined.py           -> ice_fraction/runner.py
ssfr_R1_analysis.py                 -> r1_analysis/workflow.py
R1_flag_check.py                    -> r1_analysis/helpers.py
R1_ssfr_zen_nad_compare.py          -> r1_analysis/plots.py
ssfr_SI_plot.py                     -> si_plots/plots.py
```

## Suggested Order Of Work

1. Move only the SSFR atmospheric-correction files into `lrt_sim/ssfr_atm_corr/`.
2. Update relative imports and verify:
   ```bash
   python3 -m py_compile lrt_sim/ssfr_atm_corr/*.py
   python3 -m lrt_sim.ssfr_atm_corr.runner
   ```
3. Move original backup files into `lrt_sim/legacy/`.
4. Move product-processing files after atmospheric correction is stable.
5. Move CRE, ERA5 albedo, ice fraction, R1 analysis, and SI plotting after checking their imports and shared helpers.

## Notes

- `ssfr_atm_corr_ori.py` must be preserved as a backup.
- Legacy atmospheric-correction processing and plotting scripts are preserved in
  `legacy/`, while the active native/final-extension product path is under
  `ssfr_atm_corr/`.
- Avoid changing scientific logic during file moves.
- Prefer compatibility import shims only if old run commands need to keep working temporarily.
