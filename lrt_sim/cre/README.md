# CRE workflow — surface albedo → cloud radiative effect

This package computes the **surface cloud radiative effect (CRE)** for ARCSIX
cases. The core idea is to take the atmospheric-corrected surface albedo from
different flight times/cases and compute the CRE **under the same cloud input**,
so you can isolate how surface albedo controls the surface CRE.

The workflow has two stages:

1. **Build albedo files** — write each case's atmospheric-corrected, extended
   (300–4000 nm) surface albedo to a `.dat` file libRadtran can read.
2. **Run + plot CRE** — for one fixed cloud case, sweep cloud water path (CWP)
   and solar zenith angle (SZA) in libRadtran, optionally looping over the
   albedo files from stage 1, then post-process into CRE curves and figures.

```
ssfr_atm_corr pipeline                 cre package
─────────────────────                  ───────────
processing_runner ─► combined ─►  sfc_alb_combined_spring_summer.pkl
                                          │
                                          ▼
                                  cre_sim (manual_alb=None)
                                          │  extend + clip albedo
                                          ▼
                              data/sfc_alb_cre/*_cre_alb.dat      ◄─ STAGE 1 (albedo files)
                                          │
                                          ▼
                          cre_sim (manual_alb=<albedo .dat list>)
                                          │  libRadtran SW + LW sweep
                                          ▼
                  data/lrt/<case>/ssfr_simu_flux_*_cre_{sw,lw}_sza_*.csv   ◄─ STAGE 2 (raw flux)
                                          │
                                          ▼
                                     cre_plot
                                          │  SW/LW/net CRE, zero crossings
                                          ▼
                  data/lrt/<case>/<date>_<case_tag>_cre_simulations_*.csv + figures
```

> **Run everything from the `lrt_sim/` directory.** Paths are resolved relative
> to `../data` (see `ssfr_atm_corr/settings.py`). Use the `er3t_env` conda
> environment.

---

## Prerequisites

1. The combined atmospheric-correction product must exist:
   `data/sfc_alb_combined/sfc_alb_combined_spring_summer.pkl`
   (produced by `python -m lrt_sim.ssfr_atm_corr.processing_runner` then the
   combine step in `ssfr_atm_corr/combined.py`).
2. Gas absorption files for each date (run `arcsix_gas_insitu.py` first).
3. Per-leg cloud-observation pickles in `data/flt_cld_obs_info/` — still needed
   for the MARLI water-vapor profile and solar azimuth, which are not in the
   combined product. (Surface albedo, SZA and KT19 come from the combined
   product via `load_case_from_combined`.)

---

## Stage 1 — build the albedo files

Each `.dat` is the **mean of the canonical extended albedo** taken directly from
the combined product. The combined product already carries the atmospheric-
corrected albedo on the 300–4000 nm grid (`alb_atm_corrected_ext_{season}_all`),
so `cre_sim` **does not re-extend** — it just averages the selected rows and
clips to [0, 1]. (It falls back to `alb_extention(...)` only when the combined
product has no extended albedo, e.g. an old pickle or the per-leg fallback.)

### Recommended: the albedo-case catalog (`ext_alb_cases.py`)

`cre/ext_alb_cases.py` is a **declarative catalog** of albedo cases — one entry
per `.dat` file, giving a date, a flight time window, an altitude (for the
filename) and an optional scale factor. The generator loads the combined product
once and writes every file:

```bash
cd lrt_sim
conda run -n er3t_env python -m cre.ext_alb_cases            # dry-run preview
conda run -n er3t_env python -m cre.ext_alb_cases --apply    # write .dat (+ backup) + CSV
conda run -n er3t_env python -m cre.ext_alb_cases --csv-only # only refresh the CSV
```

Generating also writes `data/sfc_alb_cre/ext_alb_broadband.csv` summarizing each
file's **solar-flux-weighted broadband albedo** (weight =
`arcsix_ssfr_solar_flux_slit.dat`), with columns `filename, date, t0, t1,
alt_km, scale, n_rows, broadband_albedo`.

To add an albedo file, append an entry to `EXT_ALB_CASES`:

```python
EXT_ALB_CASES = [
    {'date': '20240725', 'time_range': (15.881, 15.903), 'alt': 0.33},
    {'date': '20240528', 'time_range': (15.610, 17.404), 'alt': 0.22, 'scale': 0.99},
    ...
]
```

The output name is `sfc_alb_{date}_{t0:.3f}_{t1:.3f}_{alt:.2f}km_cre_alb[_scale_{f}X].dat`,
matching the names referenced by `MANUAL_ALB_SWEEP`. Entries whose date/window
are not in the combined product are skipped with a warning.

### Ad-hoc, in Python (no libRadtran)

For one-off albedo files, `cre_sim` exposes the underlying helpers:

```python
from cre.cre_sim import build_albedo_file, mean_extended_albedo

# a flight time window (decimal-hour UTC) within a date
build_albedo_file('../data/sfc_alb_cre/my_window.dat',
                  '20240603', time_range=(14.711, 14.789))

# whole case by exact case_tag
build_albedo_file('../data/sfc_alb_cre/my_case.dat',
                  '20240603', case_tag='cloudy_atm_corr_2')

# just the spectrum (ext_wvl, mean_albedo), no file
ext_wvl, alb = mean_extended_albedo('20240725', time_range=(15.88, 15.91))
```

`select_combined_rows(...)` is the underlying selector, `mean_extended_albedo(...)`
averages the extended albedo over the selection, and `build_albedo_file(...)`
writes the `.dat`. All select straight from the combined product's extended grid.

### As a side effect of a CRE run

When `cre_sim` runs with `manual_alb=None` it also writes that case's albedo to
`data/sfc_alb_cre/sfc_alb_{date}_{t0}_{t1}_{alt}km_cre_alb.dat` (same extended
albedo as the catalog). So `python -m cre.cre_runner --case-id case_004 --mode sw`
produces `case_004`'s albedo as a by-product.

---

## Stage 2 — calculate CRE

### Option A — a case under its own albedo

```bash
cd lrt_sim
conda run -n er3t_env python -m cre.cre_runner --case-id case_004 --mode both
```

This runs libRadtran for shortwave (`--mode sw`), longwave (`--mode lw`), or both
(`--mode both`), sweeping the CWP/COT list and SZA grid. Raw output:
`data/lrt/<date>_<case_tag>_sat_cloud/ssfr_simu_flux_*_cre_{sw,lw}_sza_*.csv`.

### Option B — one cloud case under many surface albedos (the albedo sweep)

This is the "different albedo, same cloud" comparison. The target cloud case is
re-run once per albedo `.dat` from stage 1:

```bash
cd lrt_sim
conda run -n er3t_env python -m cre.cre_runner --case-id case_004 --mode both --manual-alb-sweep
```

`--manual-alb-sweep` passes `cre_cases.MANUAL_ALB_SWEEP` (a list of `.dat`
filenames in `data/sfc_alb_cre/`) as `manual_alb`. Each entry produces a
separate `..._alb-manual-<name>_*.csv`. **Every file in `MANUAL_ALB_SWEEP` must
already exist** (stage 1) — `cre_sim` raises `FileNotFoundError` otherwise.

### SZA chunking note

`cre_sim`'s active `sza_arr` is intentionally a small chunk (e.g. `[50, 52.5]`)
so the cluster runs can be batched; the other chunks are commented just above it.
Run the chunks you need so the SZA grid covers what `cre_plot` reads
(`[50 … 75]` including the case-mean SZA).

---

## Stage 3 — post-process and plot

`cre_plot` reads the per-SZA flux CSVs, converts to SW / LW / net CRE vs CWP,
finds the net-CRE zero crossing, and writes summary products + figures.

```bash
cd lrt_sim
conda run -n er3t_env python cre/cre_plot.py        # runs plot_cre_case(...) for the example case
```

Outputs in `data/lrt/<date>_<case_tag>_sat_cloud/`:

- `<date>_<case_tag>_cre_simulations_all_alb.csv` — full CWP grid
- `<date>_<case_tag>_cre_simulations_real_cases_all_alb.csv` — flight-observed CWP points
- `<date>_<case_tag>_cre_simulations_alb_spectra.pkl` — albedo spectra + broadband values

Figures go to `fig/<date>/`.

To plot a different case or albedo set, call `plot_cre_case(config, case_id,
manual_alb=[...])` from `cre_plot.py`.

---

## Reference: where things live

| Item | Path |
|------|------|
| Combined albedo product (input) | `data/sfc_alb_combined/sfc_alb_combined_spring_summer.pkl` |
| Per-case albedo files (stage 1) | `data/sfc_alb_cre/*_cre_alb.dat` |
| Raw CRE flux CSVs (stage 2) | `data/lrt/<date>_<case_tag>_{clear,sat_cloud}/ssfr_simu_flux_*_cre_*.csv` |
| CRE summary CSV/pkl (stage 3) | `data/lrt/<date>_<case_tag>_*/<date>_<case_tag>_cre_simulations_*` |
| Figures | `fig/<date>/`, `fig/<case_tag>/` |

| Module | Role |
|--------|------|
| `cre_sim.py` | `cre_sim()` / `process_cre_case()` — albedo + libRadtran CRE sweep; albedo selectors (`select_combined_rows`, `mean_extended_albedo`, `build_albedo_file`) |
| `cre_plot.py` | `cre_sim_plot()` / `plot_cre_case()` — CRE post-processing + figures |
| `ext_alb_cases.py` | declarative albedo-case catalog + generator for `data/sfc_alb_cre/*.dat` |
| `cre_cases.py` | CRE-only config: case ids, SZA grid, CWP sweep, `MANUAL_ALB_SWEEP` |
| `cre_runner.py` | CLI entry point |

Base case parameters (date, time ranges, levels, cloud microphysics,
clear/cloudy) come from the shared `ssfr_atm_corr.case_catalog`; the surface
albedo comes from the combined product.

---

## Known caveat

A standalone, libRadtran-free albedo builder now exists
(`build_albedo_file` / `mean_extended_albedo`, see stage 1), and the albedo is
the canonical extended product, so the extension is consistent with the rest of
the pipeline.

What remains: the **`cre_sim` side-effect filenames** embed the time window and
altitude (`{t0}_{t1}_{alt}km`), derived from the combined-product rows. If those
rows shift, a regenerated file gets a different name and no longer matches the
hardcoded strings in `MANUAL_ALB_SWEEP` → `FileNotFoundError`. To avoid this,
build the sweep files yourself with `build_albedo_file(out_path, ...)` using a
stable `out_path` of your choosing, and point `manual_alb` at those names.
