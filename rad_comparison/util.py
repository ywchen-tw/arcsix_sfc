import os
import sys
import glob
import datetime
from dataclasses import dataclass
from enum import IntFlag, auto
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import er3t
from netCDF4 import Dataset
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


# --- Configuration ----------------------------------------------------------

@dataclass(frozen=True)
class FlightConfig:
    mission: str
    platform: str
    data_root: Path
    root_mac: Path
    root_linux: Path

    def hsk(self, date_s):    return f"{self.data_root}/{self.mission}-HSK_{self.platform}_{date_s}_v0.h5"
    def ssfr(self, date_s):   return f"{self.data_root}/{self.mission}-SSFR_{self.platform}_{date_s}_R1.h5"
    def ssrr(self, date_s):   return f"{self.data_root}/{self.mission}-SSRR_{self.platform}_{date_s}_R0.h5"
    def hsr1(self, date_s):   return f"{self.data_root}/{self.mission}-HSR1_{self.platform}_{date_s}_R0.h5"
    def logic(self, date_s):  return f"{self.data_root}/{self.mission}-LOGIC_{self.platform}_{date_s}_RA.h5"
    def sat_coll(self, date_s): return f"{self.data_root}/{self.mission}-SAT-CLD_{self.platform}_{date_s}_v0.h5"
    def marli(self, date_s):
        root = self.root_mac if sys.platform == "darwin" else self.root_linux
        return f"{root}/marli/ARCSIX-MARLi_P3B_{date_s}_R0.cdf"
    def kt19(self, date_s):
        root = self.root_mac if sys.platform == "darwin" else self.root_linux
        return f"{root}/kt19/ARCSIX-MetNav-KT19-10Hz_P3B_{date_s}_R0.ict"
    def sat_nc(self, date_s, raw):
        root = self.root_mac if sys.platform == "darwin" else self.root_linux
        return f"{root}/sat-data/{date_s}/{raw}"


# --- SSFR data quality flags ------------------------------------------------

class ssfr_flags(IntFlag):
    """Bitmask flags for SSFR data quality filtering.

    Each flag marks a condition that may corrupt the measurement.
    Test with ``(data['flag'] & ssfr_flags.<name>) != 0``.
    """
    pitch_roll_exceed_threshold = auto()  # pitch or roll angle exceeded threshold
    camera_icing = auto()                 # camera icing at time of measurement
    camera_icing_pre = auto()             # camera icing within 1 hour before measurement
    zen_toa_over_threshold = auto()       # zenith TOA irradiance exceeds upper bound
    alp_ang_pit_rol_issue = auto()        # leveling platform angle exceeded threshold


# --- Helpers ----------------------------------------------------------------

def load_h5(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return er3t.util.load_h5(str(path))


def nearest_indices(t_hsk, mask, times):
    """Vectorized nearest-index lookup: for each HSK time in the leg mask,
    find the closest index in `times`."""
    return np.argmin(np.abs(times[:, None] - t_hsk[mask][None, :]), axis=0)


def lonlat_dist(ref_lon, ref_lat, xy_lon, xy_lat):
    """Haversine great-circle distance (km) between a reference point and array of points."""
    R = 6373.0
    xy_lat_r  = np.radians(xy_lat)
    xy_lon_r  = np.radians(xy_lon)
    ref_lat_r = np.radians(ref_lat)
    ref_lon_r = np.radians(ref_lon)
    dlon = xy_lon_r - ref_lon_r
    dlat = xy_lat_r - ref_lat_r
    a = np.sin(dlat / 2)**2 + np.cos(ref_lat_r) * np.cos(xy_lat_r) * np.sin(dlon / 2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# --- Solar flux -------------------------------------------------------------

def solar_interpolation_func(solar_flux_file, date):
    """Return an interp1d function for the solar flux spectrum, scaled to the given date.

    Reads a whitespace-delimited solar flux file (wavelength in nm, flux in mW/m²/nm),
    converts to W/m²/nm, applies the Earth-Sun distance correction, and returns a
    linear interpolator over wavelength.
    """
    f_solar = pd.read_csv(solar_flux_file, sep=r'\s+', comment='#', names=['wvl', 'flux'])
    wvl_solar = f_solar['wvl'].values
    flux_solar = f_solar['flux'].values / 1000.0  # mW → W /m²/nm
    flux_solar *= er3t.util.cal_sol_fac(date)
    return interp1d(wvl_solar, flux_solar, bounds_error=False, fill_value=0.0)


# --- AVIRIS utilities -------------------------------------------------------

def load_aviris_rdn(aviris_dir, date_s, aviris_id=None):
    """Find and load AVIRIS RDN (L1B radiance) NetCDF file.

    Returns (wvl_nm, rad_W_nm_m2_sr, lon_2d, lat_2d) or None if not found.
    Radiance is converted from uW nm-1 cm-2 sr-1 to W nm-1 m-2 sr-1.
    """
    pattern = f'{aviris_id}*.nc' if aviris_id else 'ang*.nc'
    files = sorted(glob.glob(os.path.join(aviris_dir, pattern)))
    rdn_file = next(
        (f for f in files if date_s in os.path.basename(f) and 'RDN' in os.path.basename(f)),
        None
    )
    if rdn_file is None:
        print(f"No AVIRIS RDN file found for {date_s} in {aviris_dir}")
        return None
    print(f"Loading AVIRIS RDN: {rdn_file}")
    with Dataset(rdn_file) as ds:
        wvl = np.array(ds.groups['radiance'].variables['wavelength'][:], dtype=float)
        rad = ds.groups['radiance'].variables['radiance'][:] * 1e-6 * 1e4  # → W nm-1 m-2 sr-1
        lon = ds.variables['lon'][:]
        lat = ds.variables['lat'][:]
    return wvl, rad, lon, lat


def extract_aviris_spectrum(wvl, rad, lon, lat, lon_mean, lat_mean,
                             half_width: 'int | list[int]' = 4, wvl_nancheck=550.0):
    """Extract mean radiance spectrum near one or many (lon_mean, lat_mean) points.

    lon_mean / lat_mean may each be a scalar or a 1-D array of length n_q.
    half_width may be a single int or a list of ints.
    Distance to the AVIRIS grid is computed once per query point.
    Valid-pixel fallback arrays are cached and reused across query points.

    Returns:
        scalar lonlat, scalar hw → (spectrum,        unc,        nearest_lon,  nearest_lat)
        scalar lonlat, list   hw → (list_of_spectra, list_of_uncs, nearest_lon,  nearest_lat)
        array  lonlat, scalar hw → (spectra,         uncs,         nearest_lons, nearest_lats)  # (n_q, n_wvl)
        array  lonlat, list   hw → (list_of_spectra, list_of_uncs, nearest_lons, nearest_lats)  # each (n_q, n_wvl)
    """
    scalar_lonlat = np.isscalar(lon_mean)
    lon_means = np.atleast_1d(np.asarray(lon_mean, dtype=float))
    lat_means = np.atleast_1d(np.asarray(lat_mean, dtype=float))
    n_q = len(lon_means)

    scalar_hw = isinstance(half_width, (int, np.integer))
    half_widths: list[int] = [half_width] if scalar_hw else list(half_width)  # type: ignore[assignment]
    n_hw = len(half_widths)
    n_wvl = rad.shape[0]

    nancheck_idx = np.argmin(np.abs(wvl - wvl_nancheck))
    valid_mask = ~np.isnan(rad[nancheck_idx, :, :])   # (n_rows, n_cols)

    all_spectra = np.full((n_hw, n_q, n_wvl), np.nan)
    all_uncs    = np.full((n_hw, n_q, n_wvl), np.nan)
    near_lons   = np.full(n_q, np.nan)
    near_lats   = np.full(n_q, np.nan)

    # Build KD-tree once from flattened AVIRIS grid — O(N log N), queried in O(n_q log N)
    _lon_flat = lon.ravel()
    _lat_flat = lat.ravel()
    _tree = cKDTree(np.column_stack([_lon_flat, _lat_flat]))
    _, _flat_idxs = _tree.query(np.column_stack([lon_means, lat_means]))  # (n_q,)
    _min_idxs = [np.unravel_index(i, lon.shape) for i in _flat_idxs]

    fallback_pixels = None   # (v_lon, v_lat, v_rad) — cached on first use
    fallback_tree   = None

    for k in range(n_q):
        min_idx = _min_idxs[k]
        near_lons[k] = float(lon[min_idx])
        near_lats[k] = float(lat[min_idx])

        for ih, h in enumerate(half_widths):
            i0 = max(0, min_idx[0] - h)
            i1 = min(rad.shape[1], min_idx[0] + h + 1)
            j0 = max(0, min_idx[1] - h)
            j1 = min(rad.shape[2], min_idx[1] + h + 1)
            subset = rad[:, i0:i1, j0:j1]
            if np.all(np.isnan(subset)):
                print(f"Window (q={k}, half_width={h}) all-NaN, using nearest valid pixels...")
                if fallback_pixels is None:
                    v_lon = lon[valid_mask]
                    v_lat = lat[valid_mask]
                    fallback_pixels = (v_lon, v_lat, rad[:, valid_mask])
                    fallback_tree   = cKDTree(np.column_stack([v_lon, v_lat]))
                v_lon, v_lat, v_rad = fallback_pixels
                _, top25 = fallback_tree.query([lon_means[k], lat_means[k]], k=25)  # type: ignore[union-attr]
                subset = v_rad[:, top25]
                near_lons[k] = float(v_lon.ravel()[top25[0]])
                near_lats[k] = float(v_lat.ravel()[top25[0]])
                print(f"  Nearest valid: lon={near_lons[k]:.4f}, lat={near_lats[k]:.4f}")
            reduce_axes = tuple(range(1, subset.ndim))
            all_spectra[ih, k, :] = np.nanmean(subset, axis=reduce_axes)
            all_uncs[ih, k, :]    = np.nanstd(subset,  axis=reduce_axes)

    if scalar_lonlat:
        spectra = [all_spectra[ih, 0, :] for ih in range(n_hw)]
        uncs    = [all_uncs[ih, 0, :]    for ih in range(n_hw)]
        out_lon, out_lat = float(near_lons[0]), float(near_lats[0])
    else:
        spectra = [all_spectra[ih, :, :] for ih in range(n_hw)]   # each (n_q, n_wvl)
        uncs    = [all_uncs[ih, :, :]    for ih in range(n_hw)]
        out_lon, out_lat = near_lons, near_lats

    if scalar_hw:
        return spectra[0], uncs[0], out_lon, out_lat
    return spectra, uncs, out_lon, out_lat


# --- Generic plot helpers ---------------------------------------------------

def make_color_series(values):
    """Map scalar array to RGBA colors using jet colormap."""
    norm = mcolors.Normalize(vmin=np.nanmin(values), vmax=np.nanmax(values))
    return cm.ScalarMappable(norm=norm, cmap=cm.jet).to_rgba(values)


def savefig(fig, date_s, case_tag, suffix):
    """Save figure to fig/{date_s}/{date_s}_{case_tag}_{suffix}.png"""
    fig.savefig(f'fig/{date_s}/{date_s}_{case_tag}_{suffix}.png',
                bbox_inches='tight', dpi=150)
