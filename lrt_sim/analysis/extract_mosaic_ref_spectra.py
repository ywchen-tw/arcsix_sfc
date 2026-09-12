"""Extract representative MOSAiC surface-type spectra (Smith et al. 2021 dataset).

Reads the raw per-line survey CSVs downloaded from the Arctic Data Center
(doi:10.18739/A2FT8DK8Z) in ``data/SI_data/`` and writes
``data/SI_data/mosaic_ref_spectra.csv`` — the four surface-type spectra
(sites documented in Light et al. 2022, Figs 2/8) that
``ssfr_vs_sheba_mosaic.py`` overlays in its panel (a).

Run: python analysis/extract_mosaic_ref_spectra.py   (any cwd)
"""
import csv
import os

import numpy as np

_SI_DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data', 'SI_data')

def parse(fn):
    rows = list(csv.reader(open(fn)))
    pos = types = None
    wl_start = None
    ncol = 0
    for i, r in enumerate(rows):
        key = r[0].strip() if r else ''
        if key == 'Position':
            pos = [c.strip() for c in r[1:]]
        elif key == 'Surface type':
            types = [c.strip() for c in r[1:]]
        elif key == 'Wavelengths':
            wl_start = i + 1
            ncol = len(r) - 1
            break
    wvl, alb = [], []
    for r in rows[wl_start:]:
        if not r or not r[0].strip():
            continue
        try:
            w = float(r[0])
        except ValueError:
            continue
        wvl.append(w)
        vals = [float(c) if c.strip() and c.strip() != 'NaN' else np.nan
                for c in r[1:ncol + 1]]
        vals += [np.nan] * (ncol - len(vals))
        alb.append(vals)
    return pos, types, np.array(wvl), np.array(alb)

targets = [
    (f'{_SI_DATA}/mosaic_spec_20200620_LDL.csv', '165', 'melting_snow_0620_LD165'),
    (f'{_SI_DATA}/mosaic_spec_20200724_LDL.csv', '165', 'bare_ice_0724_LD165'),
    (f'{_SI_DATA}/mosaic_spec_20200724_LDL.csv', '175', 'dark_pond_0724_LD175'),
    (f'{_SI_DATA}/mosaic_spec_20200917_KINDER.csv', '20', 'autumn_snow_0917_K20'),
]
out = {}
wvl_ref = None
for fn, p, name in targets:
    pos, types, wvl, alb = parse(fn)
    pos_f = [float(x) if x else np.nan for x in pos]
    j = int(np.nanargmin(np.abs(np.array(pos_f) - float(p))))
    print(f'{name}: pos {p} type={types[j]}  wvl {wvl.min():.0f}-{wvl.max():.0f} '
          f'n={len(wvl)}  alb@500={alb[np.argmin(abs(wvl - 500)), j]:.3f} '
          f'@1064={alb[np.argmin(abs(wvl - 1064)), j]:.3f}')
    if wvl_ref is None:
        wvl_ref = wvl
        out[name] = alb[:, j]
    else:
        out[name] = np.interp(wvl_ref, wvl, alb[:, j], left=np.nan, right=np.nan)

with open('/Users/yuch8913/programming/arcsix_sfc/data/SI_data/mosaic_ref_spectra.csv', 'w') as f:
    f.write('# Representative MOSAiC surface-type spectral albedos, extracted from\n')
    f.write('# Smith et al. (2021), Arctic Data Center doi:10.18739/A2FT8DK8Z\n')
    f.write('# (sites documented in Light et al. 2022, Figs 2/8): melting snow =\n')
    f.write('# 2020-06-20 Lemon Drop 165 m; bare melting ice = 2020-07-24 LD 165 m;\n')
    f.write('# dark pond = 2020-07-24 LD 175 m; autumn snow = 2020-09-17 Kinder 20 m.\n')
    f.write('wvl_nm,' + ','.join(out.keys()) + '\n')
    for i, w in enumerate(wvl_ref):
        f.write(f'{w:.0f},' + ','.join(
            f'{out[k][i]:.4f}' if np.isfinite(out[k][i]) else '' for k in out) + '\n')
print('written mosaic_ref_spectra.csv, n_wvl =', len(wvl_ref))
