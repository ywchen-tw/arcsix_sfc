"""
Test for cloud profile file generation — replicates the fname_cld logic from
arcsix_lw_rad_test.py (lines ~1138-1153) without importing er3t.

Acknowledgement
---------------
get_cld_cfg() and gen_cloud_1d() are adapted from EaR³T
(Education and Research 3D Radiative Transfer Toolbox):

  Authors : Vikas Nataraja, Yu-Wen Chen, Ken Hirata, Hong Chen, Sebastian Schmidt
  GitHub  : https://github.com/hong-chen/er3t
  Paper   : Chen et al. (2023), AMT, doi:10.5194/amt-16-1971-2023
  License : GNU GPLv3

Original sources:
  er3t.rtm.lrt.lrt_cfg.get_cld_cfg   → er3t/rtm/lrt/lrt_cfg.py
  er3t.rtm.lrt.util.gen_cloud_1d     → er3t/rtm/lrt/util.py

The file format (altitude / LWC / CER) is identical for water and ice clouds.
The cloud_type only affects the libRadtran input directive used downstream
(wc_file + wc_properties for water; ic_file + ic_properties for ice).

libRadtran wc_file / ic_file layer-quantity convention (since v1.4)
--------------------------------------------------------------------
Each row defines the layer from that altitude UP TO the next altitude above it:

  altitude [km]   LWC [g/m³]   CER [µm]
  3.0000          0.0000        0.0000   ← cloud top: LWC=0 marks the upper boundary
  2.0000          0.0500       25.0000   ← layer 2–3 km is cloudy (LWC > 0)
  0.0000          0.0000        0.0000   ← ground level (always appended if missing)

Rules:
  • The topmost row MUST have LWC=0 to define the cloud top (no cloud above it).
  • A layer is cloudy when the row at its base altitude has LWC > 0.
  • gen_cloud_1d() sorts altitudes descending, sets lwc[0]=0 (top boundary),
    and fills lwc[1:] with the computed LWC — correctly implementing this rule.

make_cld_cfg() supports three altitude modes via levels / snap arguments:
  Option 1  (levels=None)            : exact [cbh, cth], no grid needed
  Option 2a (levels=..., snap=True)  : snap cbh/cth to nearest grid levels (bisect)
  Option 2b (levels=..., snap=False) : levels provided but exact [cbh, cth] kept

Output files in output/test_cld_profiles/:
  cld_{ice,water}_no_levels.txt  — Option 1
  cld_{ice,water}_snap_grid.txt  — Option 2a
  cld_{ice,water}_exact_grid.txt — Option 2b
"""

import os
import bisect
import numpy as np


# ---------------------------------------------------------------------------
# Standalone helpers (replicate er3t.rtm.lrt.get_cld_cfg / gen_cloud_1d)
# ---------------------------------------------------------------------------

def get_cld_cfg():
    """Return a default cloud configuration dictionary.

    Matches er3t.rtm.lrt.lrt_cfg.get_cld_cfg() exactly.
    Note: ic_properties has no default in er3t — set it explicitly for ice clouds.
    """
    return {
        'cloud_file'              : 'LRT_cloud_profile.txt',
        'cloud_optical_thickness' : 20.0,
        'cloud_effective_radius'  : 10.0,
        'liquid_water_content'    : 0.02,
        'cloud_type'              : 'water',   # 'water' or 'ice'
        'wc_properties'           : 'mie',     # used when cloud_type='water'
        # ic_properties is required when cloud_type='ice', e.g. 'yang2013'
        'cloud_altitude'          : np.arange(0.9, 1.31, 0.1),
    }


def gen_cloud_1d(cld_cfg):
    """Write a 1-D cloud profile text file from *cld_cfg* (in-place update).

    The three-column format (altitude / LWC / CER) is the same for both
    water and ice clouds. Mirrors er3t.rtm.lrt.util.gen_cloud_1d exactly.

    libRadtran layer-quantity rule (default since v1.4):
      - Altitudes are written top-down (descending).
      - The top row has LWC=CER=0: this marks the cloud-top boundary;
        no cloud exists above it.
      - Each subsequent row's LWC/CER applies to the layer from that
        altitude up to the row above it.
      - A trailing 0.0 km row (LWC=CER=0) is appended if the lowest
        altitude is not already at ground level.
    """
    cld_cfg['cloud_file'] = os.path.abspath(cld_cfg['cloud_file'])

    altitude = cld_cfg['cloud_altitude']
    alt = np.sort(altitude)[::-1]          # descending: cloud top first
    lwc = np.zeros_like(alt, dtype=float)
    cer = np.zeros_like(alt, dtype=float)

    lwc_val = cld_cfg['liquid_water_content']
    cer_val = cld_cfg['cloud_effective_radius']

    # lwc[0] / cer[0] remain 0 — cloud-top boundary row (required by libRadtran)
    if not isinstance(lwc_val, np.ndarray):
        lwc[1:] = lwc_val
        cer[1:] = cer_val
    else:
        lwc[1:] = lwc_val[::-1]
        cer[1:] = cer_val[::-1]

    os.makedirs(os.path.dirname(cld_cfg['cloud_file']), exist_ok=True)

    with open(cld_cfg['cloud_file'], 'w') as f:
        f.write('# Altitude[km]    Liquid Water Content [g/m3]    Cloud Effective Radius [um]\n')
        for i in range(len(alt)):
            f.write('     %.4f                   %.4f                          %.4f\n'
                    % (alt[i], lwc[i], cer[i]))
        # append ground-level row (LWC=CER=0) if lowest altitude is not 0
        if abs(alt[-1] - 0.0) > 0.001:
            f.write('     %.4f                   %.4f                          %.4f\n'
                    % (0.0, 0.0, 0.0))

    return cld_cfg


# Bulk densities used in the two-stream CWP–COT relationship
RHO_WATER = 1000.0   # liquid water density [kg/m³]
RHO_ICE   =  917.0   # bulk ice density     [kg/m³]


def calc_cwp(cot, cer, cloud_type):
    """Derive cloud water path [kg/m²] from optical thickness and effective radius.

    Two-stream approximation (van de Hulst / Stephens):
        τ = (3/2) × CWP / (ρ × r_eff)
        → CWP = (2/3) × τ × ρ × r_eff

    Parameters
    ----------
    cot        : cloud optical thickness (unitless)
    cer        : cloud effective radius [µm]
    cloud_type : 'water' or 'ice'  — selects bulk density

    Returns
    -------
    cwp : cloud water path [kg/m²]
    """
    rho = RHO_WATER if cloud_type == 'water' else RHO_ICE
    return (2.0 / 3.0) * cot * rho * (cer * 1e-6)   # cer: µm → m


def print_cld_cfg(cld_cfg):
    ctype = cld_cfg['cloud_type']
    props_key = 'ic_properties' if ctype == 'ice' else 'wc_properties'
    print(f'  cloud_file              : {cld_cfg["cloud_file"]}')
    print(f'  cloud_type              : {ctype}')
    print(f'  {props_key:24s}: {cld_cfg[props_key]}')
    print(f'  cloud_altitude          : {cld_cfg["cloud_altitude"]} km')
    print(f'  cloud_effective_radius  : {cld_cfg["cloud_effective_radius"]} µm')
    lwc = cld_cfg['liquid_water_content']
    cgt = cld_cfg['cloud_altitude'][-1] - cld_cfg['cloud_altitude'][0]  # km (sorted asc)
    cwp = lwc * abs(cgt) * 1000.0 / 1000.0  # g/m³ × km×1000m/km / 1000 → kg/m²
    print(f'  liquid_water_content    : {lwc:.4f} g/m³  (CWP ≈ {cwp:.6f} kg/m²)')
    print(f'  cloud_optical_thickness : {cld_cfg["cloud_optical_thickness"]}')



def make_cld_cfg(fname, cloud_type, cbh, cth, cwp, cot, cer,
                 levels=None, snap=True):
    """Build and return a cloud configuration dictionary.

    Parameters
    ----------
    fname       : output cloud profile filename
    cloud_type  : 'ice' or 'water'
    cbh, cth    : cloud base / top height [km]
    cwp         : cloud water path [kg/m²]
    cot         : cloud optical thickness
    cer         : cloud effective radius [µm]
    levels      : altitude grid (numpy array, [km]).
                  If None → always use exact [cbh, cth] (option 1).
                  If provided → behaviour controlled by *snap*.
    snap        : only used when levels is provided.
                  True  → snap cbh/cth to nearest grid levels (option 2a).
                  False → use exact [cbh, cth] even though levels was given (option 2b).
    """
    cgt = cth - cbh
    lwc = cwp * 1000.0 / (cgt * 1000.0)   # kg/m² → g/m³

    if levels is None:
        # Option 1: no grid — use exact heights
        altitude = np.array([cbh, cth])
    else:
        if snap:
            # Option 2a: snap to grid levels that bracket [cbh, cth]
            #   cbh: bisect_right - 1  → floor to the level AT or below cbh
            #   cth: bisect_left       → ceiling to the level AT or above cth
            # Example: cbh=2.0, cth=2.5 on grid [...,2.0,3.0,...]
            #   bisect_right(2.0)-1 = 4 → 2.0  (exact match kept)
            #   bisect_left(2.5)    = 5 → 3.0  (snapped up to next level)
            #   → altitude = [2.0, 3.0]  (layer 2–3 km is cloudy)
            cbh_ind = bisect.bisect_right(levels, cbh) - 1
            cth_ind = bisect.bisect_left(levels, cth)
            altitude = levels[cbh_ind:cth_ind + 1]
        else:
            # Option 2b: levels provided but use exact heights anyway
            altitude = np.array([cbh, cth])

    cfg = get_cld_cfg()
    cfg['cloud_file']              = fname
    cfg['cloud_type']              = cloud_type
    cfg['cloud_altitude']          = altitude
    cfg['cloud_effective_radius']  = cer
    cfg['liquid_water_content']    = lwc
    cfg['cloud_optical_thickness'] = cot
    if cloud_type == 'ice':
        cfg['ic_properties'] = 'yang2013'
    else:
        cfg['wc_properties'] = 'mie'
    return cfg


# ---------------------------------------------------------------------------
# Shared test parameters
# ---------------------------------------------------------------------------

# Altitude grid [km]
levels = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0,
                   7.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0])

fdir_out = 'output/test_cld_profiles'
os.makedirs(fdir_out, exist_ok=True)

# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------

manual_cloud_cot = 5.0    # cloud optical thickness
manual_cloud_cer = 25.0   # effective radius  [µm]
manual_cloud_cth = 2.5    # cloud top height  [km]
manual_cloud_cbh = 2.0    # cloud base height [km]

# CWP derived from COT and CER using bulk density for each cloud type
# (two-stream: CWP = 2/3 × τ × ρ × r_eff)
manual_cloud_cwp = {
    'water': calc_cwp(manual_cloud_cot, manual_cloud_cer, 'water'),
    'ice'  : calc_cwp(manual_cloud_cot, manual_cloud_cer, 'ice'),
}
print('Derived CWP:')
for ct, cwp in manual_cloud_cwp.items():
    print(f'  {ct:5s}  ρ={RHO_WATER if ct=="water" else RHO_ICE:.0f} kg/m³'
          f'  CWP = {cwp*1000:.4f} g/m²  ({cwp:.6f} kg/m²)')
print()

cases = [
    # (label,        levels,  snap,  description)
    ('no_levels',    None,    True,  'Option 1 — no levels, exact cbh/cth'),
    ('snap_grid',    levels,  True,  'Option 2a — levels provided, snap to grid'),
    ('exact_grid',   levels,  False, 'Option 2b — levels provided, keep exact cbh/cth'),
]

for label, lev, snap, desc in cases:
    print(f'=== {desc} ===')
    for cloud_type in ('ice', 'water'):
        fname = os.path.join(fdir_out, f'cld_{cloud_type}_{label}.txt')
        if os.path.exists(fname):
            os.remove(fname)
        cfg = make_cld_cfg(fname, cloud_type,
                           cbh=manual_cloud_cbh, cth=manual_cloud_cth,
                           cwp=manual_cloud_cwp[cloud_type],
                           cot=manual_cloud_cot, cer=manual_cloud_cer,
                           levels=lev, snap=snap)
        gen_cloud_1d(cfg)
        print(f'  [{cloud_type}]  cloud_altitude: {cfg["cloud_altitude"]} km')
        print_cld_cfg(cfg)
        print()

print('Done! Output written to', fdir_out)
