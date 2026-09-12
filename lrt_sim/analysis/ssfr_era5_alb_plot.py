"""Plot ERA5 forecast albedo against retrieved surface albedo, per flight date.

Standalone re-run of the ERA5 collocation figures. Instead of re-running the full
``lrt_sim.ssfr_atm_corr.combined.combined_atm_corr`` pipeline, this script reads the
cached combined product (``sfc_alb_combined_spring_summer.pkl``) -- which already stores
the ERA5 albedo collocated to each flight point -- and the ERA5 background field, then
reproduces:

    1) ``era5_alb_{date}_collocate.png`` : ERA5 albedo map + flight scatter
    2) ``bb_alb_{date}.png``            : ERA5 vs retrieved broadband albedo scatter

Map/scatter styling matches ``combined.py`` (jet, vmin=0.4, vmax=0.8). The ERA5 plots are
also emitted by ``combined.py`` during a full run; this file decouples them so they can be
regenerated from the cache alone.
"""

import os
import sys
import platform

# Anchor everything to the lrt_sim/ project dir (parent of this analysis/ folder),
# so the script runs the same regardless of the current working directory.
_BASE_DIR_ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../lrt_sim
sys.path.insert(0, _BASE_DIR_)   # make `ssfr_atm_corr` and `util` resolvable
os.chdir(_BASE_DIR_)             # keep relative paths (../data, ./fig, ./tmp) valid

if platform.system() == 'Linux':
    sys.path.append("/projects/yuch8913/arcsix_sfc/lrt_sim/")

import datetime
import logging
import pickle

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset

from ssfr_atm_corr.settings import _fdir_general_
from plot_style import apply_grl_style


def make_polar_map(lon_all, lat_all, figsize=(8, 4)):
    """NorthPolarStereo figure/ax centered on the data extent.

    Mirrors ``ssfr_atm_corr.combined.make_polar_map``; inlined here to avoid importing
    ``combined`` (which pulls in the heavy ``util`` package) for a plot-only script.
    """
    plt.close('all')
    central_lon = np.mean(lon_all)
    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=central_lon)}
    )
    ax.coastlines()
    ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black', facecolor='lightgray')
    ax.set_extent(
        [np.min(lon_all) - 2, np.max(lon_all) + 2,
         np.min(lat_all) - 2, np.max(lat_all) + 2],
        crs=ccrs.PlateCarree()
    )
    return fig, ax


def _load_era5_field():
    """Load the ERA5 forecast-albedo field and 2-D lon/lat meshes.

    Matches the reader in ``combined.py``: ``valid_time`` is days since 2024-05-01 and
    the meshgrid uses ``indexing='ij'`` so it aligns with ``fal[time, lat, lon]``.
    """
    with Dataset(f'{_fdir_general_}/era5/forecast_albedo_0_daily-mean.nc', 'r') as nc:
        era5_lon = nc.variables['longitude'][:]
        era5_lat = nc.variables['latitude'][:]
        era5_time = nc.variables['valid_time'][:]  # days since 2024-05-01
        era5_alb = nc.variables['fal'][:]          # (time, lat, lon)
    era5_time_dates = np.array([
        datetime.datetime(2024, 5, 1) + datetime.timedelta(days=int(t)) for t in era5_time
    ])
    era5_time_dates_str = np.array([t.strftime('%Y%m%d') for t in era5_time_dates])
    era5_alb = np.array(era5_alb, dtype=np.float32)
    era5_lat_mesh, era5_lon_mesh = np.meshgrid(era5_lat, era5_lon, indexing='ij')
    return era5_lon_mesh, era5_lat_mesh, era5_alb, era5_time_dates_str


def plot_era5_alb():
    """Reproduce ERA5 albedo collocation figures from the cached combined product."""
    log = logging.getLogger("era5 alb plot")
    apply_grl_style()

    combined_file = f'{_fdir_general_}/sfc_alb_combined/sfc_alb_combined_spring_summer.pkl'
    print(f"Loading combined product: {combined_file}")
    with open(combined_file, 'rb') as f:
        d = pickle.load(f)

    era5_lon_mesh, era5_lat_mesh, era5_alb, era5_time_dates_str = _load_era5_field()

    os.makedirs('./fig/ice_age', exist_ok=True)

    # Shared map extent across both seasons.
    lon_all = np.concatenate((d['lon_all_spring'], d['lon_all_summer']))
    lat_all = np.concatenate((d['lat_all_spring'], d['lat_all_summer']))

    # Accumulate per-season points for the combined scatter at the end.
    season_colors = {'spring': 'tab:blue', 'summer': 'tab:red'}
    era5_alb_by_season = {'spring': [], 'summer': []}
    bb_alb_by_season = {'spring': [], 'summer': []}

    for suffix in ('spring', 'summer'):
        lon_flight_all = d[f'lon_all_{suffix}']
        lat_flight_all = d[f'lat_all_{suffix}']
        dates_all = d[f'dates_{suffix}_all']
        era5_alb_flight_all = d[f'era5_alb_{suffix}_all']
        bb_alb_all = d[f'broadband_alb_iter2_all_{suffix}']

        era5_alb_by_season[suffix].append(era5_alb_flight_all)
        bb_alb_by_season[suffix].append(bb_alb_all)

        for date_s in sorted(set(dates_all)):
            date_s_str = str(int(date_s))
            date_mask = dates_all == date_s
            if date_mask.sum() == 0:
                continue

            era5_alb_date = era5_alb[era5_time_dates_str == date_s_str]
            if era5_alb_date.shape[0] == 0:
                log.warning("No ERA5 albedo field for %s; skipping.", date_s_str)
                continue

            lon_flight = lon_flight_all[date_mask]
            lat_flight = lat_flight_all[date_mask]
            era5_alb_flight = era5_alb_flight_all[date_mask]
            bb_alb_flight = bb_alb_all[date_mask]

            # --- ERA5 albedo map + flight scatter ---
            fig, ax = make_polar_map(lon_all, lat_all, figsize=(8, 6))
            c1 = ax.pcolormesh(era5_lon_mesh, era5_lat_mesh, era5_alb_date[0],
                               transform=ccrs.PlateCarree(), cmap='jet', vmin=0.4, vmax=0.8)
            ax.scatter(lon_flight, lat_flight, c=era5_alb_flight,
                       transform=ccrs.PlateCarree(), cmap='jet', vmin=0.4, vmax=0.8,
                       edgecolors='k')
            fig.colorbar(c1, ax=ax, label='ERA5 Forecast Albedo')
            fig.suptitle(f"{date_s_str}", fontsize=16)
            fig.savefig(f'./fig/ice_age/era5_alb_{date_s_str}_collocate.png',
                        dpi=300, bbox_inches='tight')
            plt.close(fig)

            # --- ERA5 vs retrieved broadband albedo scatter ---
            plt.close('all')
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(era5_alb_flight, bb_alb_flight, c='tab:blue', alpha=0.5)
            ax.plot([0, 1], [0, 1], 'k--', label='1:1 Line')
            ax.set_xlabel('ERA5 Forecast Albedo')
            ax.set_ylabel('Retrieved Broadband Surface Albedo')
            ax.set_title(f"{date_s_str}")
            ax.legend()
            fig.savefig(f'./fig/ice_age/bb_alb_{date_s_str}.png',
                        dpi=300, bbox_inches='tight')
            plt.close(fig)

            print(f"  {date_s_str}: {date_mask.sum()} flight points plotted.")

    # --- Combined ERA5 vs broadband albedo scatter (spring + summer) ---
    plt.close('all')
    fig, ax = plt.subplots(figsize=(8, 6))
    for suffix in ('spring', 'summer'):
        era5_vals = np.concatenate(era5_alb_by_season[suffix])
        bb_vals = np.concatenate(bb_alb_by_season[suffix])
        ax.scatter(era5_vals, bb_vals, c=season_colors[suffix], alpha=0.5,
                   label=suffix.capitalize())
    ax.plot([0, 1], [0, 1], 'k--', label='1:1 Line')
    ax.set_xlabel('ERA5 Forecast Albedo')
    ax.set_ylabel('Retrieved Broadband Surface Albedo')
    ax.set_title('ERA5 vs Retrieved Broadband Albedo')
    ax.legend()
    fig.savefig('./fig/ice_age/bb_alb_spring_summer.png',
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Combined spring+summer scatter saved to "
          "fig/ice_age/bb_alb_spring_summer.png")


if __name__ == '__main__':

    plot_era5_alb()
