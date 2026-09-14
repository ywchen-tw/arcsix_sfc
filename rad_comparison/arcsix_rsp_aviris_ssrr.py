"""
arcsix_rsp_aviris_ssrr.py
=====================
ARCSIX campaign: compare SSRR upward radiance against AVIRIS-NG L1B radiance
and RSP (Research Scanning Polarimeter) nadir radiance.

Workflow
--------
1. flt_trk_data_collect()  – collect HSK / SSFR / SSRR / RSP data for one
                             flight leg and save a .pkl file.
2. atm_corr_plot()         – load the .pkl, co-locate AVIRIS radiance to each
                             RSP footprint, compute cross-instrument comparisons,
                             and write diagnostic figures.

Prerequisites
-------------
  Run arcsix_gas_insitu.py first to generate the per-date gas files used by
  prepare_atmospheric_profile().

Dependencies
------------
  er3t, netCDF4, scipy, matplotlib, pandas, h5py
  Local: util.util (FlightConfig, load_h5, nearest_indices, …)
         util.arcsix_atm (prepare_atmospheric_profile)
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import os
import glob
import datetime
import logging
import pickle
import platform
import gc
from typing import Optional

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
from netCDF4 import Dataset
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from util import *

# ---------------------------------------------------------------------------
# Campaign / instrument identifiers and data paths
# ---------------------------------------------------------------------------
_mission_      = 'arcsix'
_platform_     = 'p3b'

_hsk_          = 'hsk'
_alp_          = 'alp'
_spns_         = 'spns-a'
_ssfr1_        = 'ssfr-a'
_ssfr2_        = 'ssfr-b'
_cam_          = 'nac'

if platform.system() == 'Darwin':
    _fdir_data_ = '/Volumes/argus/field/%s/processed' % _mission_
    _fdir_data_ = '../data/processed' 
    _fdir_general_ = '../data'
    _fdir_tmp_ = './tmp'
elif platform.system() == 'Linux':
    _fdir_data_ = "/pl/active/vikas-arcsix/yuch8913/arcsix/data/processed"
    _fdir_general_ = "/pl/active/vikas-arcsix/yuch8913/arcsix/data"
    _fdir_tmp_ = "/pl/active/vikas-arcsix/yuch8913/arcsix/tmp"

# ---------------------------------------------------------------------------
# Gaseous absorption band wavelength ranges (nm) used for shading in plots.
# O2-A band and seven H2O bands are masked to avoid contaminating albedo fits.
# ---------------------------------------------------------------------------
o2a_1_start, o2a_1_end = 748, 776
h2o_1_start, h2o_1_end = 672, 706
h2o_2_start, h2o_2_end = 705, 746
h2o_3_start, h2o_3_end = 884, 996
h2o_4_start, h2o_4_end = 1084, 1175
h2o_5_start, h2o_5_end = 1230, 1286
h2o_6_start, h2o_6_end = 1290, 1509
h2o_7_start, h2o_7_end = 1748, 2050
h2o_8_start, h2o_8_end = 2110, 2200

gas_bands = [(o2a_1_start, o2a_1_end), (h2o_1_start, h2o_1_end), (h2o_2_start, h2o_2_end),
                (h2o_3_start, h2o_3_end), (h2o_4_start, h2o_4_end), (h2o_5_start, h2o_5_end),
                (h2o_6_start, h2o_6_end), (h2o_7_start, h2o_7_end), (h2o_8_start, h2o_8_end)]


def ssfr_alb_plot(date_s, tmhr_range, wvl, alb, color_series,
                   alt_avg_all,
                   modis_bands_nm, modis_alb_legs, modis_alb_file,
                   case_tag,
                   ylabel='SSFR upward/downward ratio',
                   title='SSFR measurement',
                   suptitle='SSFR upward/downward ratio Comparison',
                   file_description='', 
                   lon_avg_all=None,
                   lat_avg_all=None,
                   aviris_file=None,
                   aviris_closest=False,
                   aviris_reflectance_wvl=None,
                   aviris_reflectance_spectrum=None,
                   aviris_reflectance_spectrum_unc=None
                   ):
    plt.close('all')
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if aviris_file is not None:
        aviris_label = 'AVIRIS Reflectance' if ((aviris_file is not None) and (aviris_closest)) else 'All AVIRIS mean'
        ax.scatter(aviris_reflectance_wvl, aviris_reflectance_spectrum, s=5, c='m', label=aviris_label, alpha=0.7) if ((aviris_file is not None)) else None
        ax.fill_between(aviris_reflectance_wvl, aviris_reflectance_spectrum-aviris_reflectance_spectrum_unc, aviris_reflectance_spectrum+aviris_reflectance_spectrum_unc, color='m', alpha=0.3) if aviris_file is not None else None
        if modis_alb_file is not None:
            ax.scatter(modis_bands_nm, modis_alb_legs, s=50, c='g', marker='*', label='MODIS Albedo', edgecolors='k')

    if lon_avg_all is not None and lat_avg_all is not None:
        ax.plot(wvl, alb, '-', label='Z=%.2fkm, lon:%.2f, lat: %.2f' % (alt_avg_all, lon_avg_all, lat_avg_all))
    else:
        ax.plot(wvl, alb, '-', label='Z=%.2fkm' % (alt_avg_all))
    if aviris_file is None and modis_alb_file is not None:
        ax.scatter(modis_bands_nm, modis_alb_legs, s=50, marker='*', edgecolors='k')
    for band in gas_bands:
        ax.axvspan(band[0], band[1], color='gray', alpha=0.3)
    
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
    # plt.grid(True)
    ax.tick_params(labelsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title, fontsize=13)
    fig.suptitle(suptitle, fontsize=16, y=0.98)
    fig.tight_layout()
    fig.savefig('fig/%s/%s_%s_%s_comparison.png' % (date_s, date_s, case_tag, file_description), bbox_inches='tight', dpi=150)


def ssfr_up_dn_ratio_plot(date_s, tmhr_range, wvl, up_dn_ratio, color_series, alt_avg_all, case_tag,
                          albedo_used='albedo used: SSFR upward/downward ratio',
                          file_suffix=''):
    plt.close('all')
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(wvl, up_dn_ratio, '-', label='Z=%.2fkm' % (alt_avg_all))
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('SSFR upward/downward ratio', fontsize=14)
    ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.tick_params(labelsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(albedo_used, fontsize=13)
    fig.suptitle(f'P3 level upward/downward ratio Comparison {date_s}', fontsize=16, y=0.98)
    fig.tight_layout()
    fig.savefig('fig/%s/%s_%s_ssfr_up_dn_ratio_comparison%s.png' % (date_s, date_s, case_tag, file_suffix), bbox_inches='tight', dpi=150)




def plot_aviris_location_map(ax, lon, lat, data_550, flight_lon, flight_lat, title,
                              colorbar_label='Radiance at 550 nm'):
    """Plot AVIRIS 2D map with flight track scatter overlay."""
    im = ax.pcolormesh(lon, lat, data_550, cmap='jet', shading='auto')
    plt.colorbar(im, ax=ax, label=colorbar_label)
    ax.scatter(flight_lon, flight_lat, color='red', marker='x',
               s=20, label='Flight Track')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title)
    ax.legend(fontsize=12)




def atm_corr_plot(date=datetime.datetime(2024, 5, 31),
                     tmhr_range=[14.10, 14.27],
                     case_tag='default',
                     config: Optional[FlightConfig] = None,
                     rsp_plot=False,
                     aviris_id=None,
                     gs_half_width=1,
                            ):
    log = logging.getLogger("atm corr spiral plot")
    log.info("Starting processing for %s", date.strftime("%Y%m%d"))

    assert config, "FlightConfig required"
    date_s = date.strftime("%Y%m%d")
    doy_s = date.timetuple().tm_yday
    print(f"Processing date: {date_s}, DOY: {doy_s}")
    
    # modis_alb collection code for the future analysis if needed
    # modis_alb_dir = f'{_fdir_general_}/modis_albedo'
    # # list all modis albedo files
    # modis_alb_files = sorted(glob.glob(os.path.join(modis_alb_dir, f'M*.nc')))
    # for fname in modis_alb_files:
    #     print("Checking modis file:", os.path.basename(fname).split('.')[1])
    #     if str(doy_s) in os.path.basename(fname).split('.')[1]:
    #         modis_alb_file = fname
    #         break
    # else:
    #     modis_alb_file = None

    # if modis_alb_file is not None:
    #     with Dataset(modis_alb_file, 'r') as ds:
    #         modis_lon = ds.variables['Longitude'][:]
    #         modis_lat = ds.variables['Latitude'][:]
    #         modis_bands = ds.variables['Bands'][:]
    #         modis_sur_alb = ds.variables['Albedo_1km'][:]
        
    #     modis_alb_legs = []
    #     modis_bands_nm = np.array([float(i) for i in modis_bands[:7]])*1000  # in nm 
          
    # print("modis_alb_file:", modis_alb_file)
    
    # # find the modis location closest to the flight leg center
    # if modis_alb_file is not None:
    #     dist = np.sqrt((modis_lon - lon_avg)**2 + (modis_lat - lat_avg)**2)
    #     min_idx = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
    #     modis_alb_leg = modis_sur_alb[min_idx[0], min_idx[1], :7]
    #     modis_alb_legs.append(modis_alb_leg)

    # if modis_alb_file is None:
    #     modis_bands_nm = None
    #     modis_alb_legs = None
        
    aviris_rdn = load_aviris_rdn(f'{_fdir_general_}/aviris_ng', date_s, aviris_id)
    if aviris_rdn is None:
        raise FileNotFoundError(f"No AVIRIS RDN file found for {date_s} in {_fdir_general_}/aviris_ng")
    aviris_rad_wvl, aviris_rad, aviris_rad_lon, aviris_rad_lat = aviris_rdn   
    rad_550_idx = np.argmin(np.abs(aviris_rad_wvl - 550.0))
    rad_550 = aviris_rad[rad_550_idx, :, :]

    
    fdir_cld_obs_info = f'{_fdir_general_}/flt_cld_obs_info'
    time_start, time_end = tmhr_range[0], tmhr_range[-1]
    fname_pkl = '%s/%s_cld_obs_info_%s_%s_%s_time_%.3f-%.3f_atm_corr.pkl' % (fdir_cld_obs_info, _mission_.lower(), _platform_.lower(), date_s, case_tag, time_start, time_end)
    with open(fname_pkl, 'rb') as f:
        cld_leg = pickle.load(f)
        
    time_all = cld_leg['time']
    alt_all = cld_leg['alt']
    lon_all = cld_leg['lon']
    lat_all = cld_leg['lat']
    heading = cld_leg['heading']
    rsp_time_all = cld_leg['rsp_time']
    rsp_lon_all = cld_leg['rsp_lon']
    rsp_lat_all = cld_leg['rsp_lat']
    rsp_rad_all = cld_leg['rsp_rad'] # in W m-2 sr-1 nm-1
    rsp_rad_norm_all = cld_leg['rsp_rad_norm']
    rsp_wvl_all = cld_leg['rsp_wvl']
    rsp_mu0 = cld_leg['rsp_mu0']
    rsp_sd = cld_leg['rsp_sd']
    sza = cld_leg['sza']
    saa = cld_leg['saa']
    ssfr_wvl_zen = cld_leg['ssfr_zen_wvl']
    ssfr_wvl_nad = cld_leg['ssfr_nad_wvl']
    ssfr_fdn_all = cld_leg['ssfr_zen']
    ssfr_fup_all = cld_leg['ssfr_nad']
    ssrr_rad_dn_wvl = cld_leg['ssrr_zen_wvl']
    ssrr_rad_dn_all = cld_leg['ssrr_zen_rad']
    ssrr_rad_up_wvl = cld_leg['ssrr_nad_wvl']
    ssrr_rad_up_all = cld_leg['ssrr_nad_rad']
    toa_all = cld_leg['ssfr_toa']
    
    alt_avg = np.nanmean(alt_all)  # in km
    lon_avg = np.nanmean(lon_all)
    lat_avg = np.nanmean(lat_all)
    
    ssfr_wvl = cld_leg['ssfr_zen_wvl']
    ssfr_550_ind = np.argmin(np.abs(ssfr_wvl - 550))
    ssfr_1600_ind = np.argmin(np.abs(ssfr_wvl - 1600))
    fdn_550_all = cld_leg['ssfr_zen'][:, ssfr_550_ind]
    fup_550_all = cld_leg['ssfr_nad'][:, ssfr_550_ind]
    fdn_1600_all = cld_leg['ssfr_zen'][:, ssfr_1600_ind]
    fup_1600_all = cld_leg['ssfr_nad'][:, ssfr_1600_ind]
        
    print(f"date_s: {date_s}, time: {time_start:.2f}-{time_end:.2f}, alt_avg: {alt_avg:.2f} km")
    log.info(f"Saved surface albedo updates to {_fdir_general_}/sfc_alb/sfc_alb_update_{date_s}_{case_tag}.pkl")

    n_t_ssrr = ssrr_rad_up_all.shape[0]
    n_t = rsp_rad_all.shape[0]

    # --- coarse-to-fine search for best lon/lat offset to maximise correlation ---
    _aviris_rad_wvl = aviris_rad_wvl   # save before it gets shadowed below
    _gs_wvl_ind = np.argmin(np.abs(_aviris_rad_wvl - 555))
    _gs_rsp_wvl_ind = np.argmin(np.abs(np.array(rsp_wvl_all) - 555))

    # Pre-compute once: 555nm slice, KD-tree over the full AVIRIS grid, and RSP 555nm radiance
    _aviris_rad_555 = aviris_rad[int(_gs_wvl_ind), :, :]   # (n_rows, n_cols)
    _aviris_shape = _aviris_rad_555.shape
    _aviris_tree = cKDTree(np.column_stack([aviris_rad_lon.ravel(), aviris_rad_lat.ravel()]))
    _r_rsp = rsp_rad_all[:, int(_gs_rsp_wvl_ind)]

    _gs_nr, _gs_nc = _aviris_shape

    def _eval_corr(d_lons, d_lats):
        """Evaluate correlation and RMSE for an array of (d_lon, d_lat) offset pairs.

        d_lons, d_lats: 1-D arrays of shape (N,)
        Returns corr_arr (N,), rmse_arr (N,)
        """
        d_lons = np.asarray(d_lons, dtype=float).ravel()
        d_lats = np.asarray(d_lats, dtype=float).ravel()
        N = len(d_lons)
        n_t = rsp_lon_all.shape[0]
        _rsp_lon0 = rsp_lon_all[:, 0]   # (n_t,)
        _rsp_lat0 = rsp_lat_all[:, 0]   # (n_t,)

        # Build shifted query points for all offsets at once: (N*n_t, 2)
        _shifted_lon = (_rsp_lon0[np.newaxis, :] + d_lons[:, np.newaxis]).ravel()
        _shifted_lat = (_rsp_lat0[np.newaxis, :] + d_lats[:, np.newaxis]).ravel()
        _, _flat_idxs = _aviris_tree.query(np.column_stack([_shifted_lon, _shifted_lat]))
        _ni, _nj = np.unravel_index(_flat_idxs, _aviris_shape)

        # Average over (2*gs_half_width+1)^2 window around each nearest pixel
        _a_sum = np.zeros(N * n_t, dtype=float)
        _a_cnt = np.zeros(N * n_t, dtype=int)
        for _di in range(-gs_half_width, gs_half_width + 1):
            for _dj in range(-gs_half_width, gs_half_width + 1):
                _wi = np.clip(_ni + _di, 0, _gs_nr - 1)
                _wj = np.clip(_nj + _dj, 0, _gs_nc - 1)
                _v = _aviris_rad_555[_wi, _wj]
                _ok = ~np.isnan(_v)
                _a_sum[_ok] += _v[_ok]
                _a_cnt[_ok] += 1
        _a = np.where(_a_cnt > 0, _a_sum / _a_cnt, np.nan).reshape(N, n_t)  # (N, n_t)

        # Vectorized correlation and RMSE across all N offsets simultaneously
        _r = _r_rsp[np.newaxis, :]                                 # (1, n_t)
        _valid = ~np.isnan(_a) & ~np.isnan(_r)                    # (N, n_t)
        n_valid = _valid.sum(axis=1).astype(float)                 # (N,)
        n_safe = np.where(n_valid > 1, n_valid, 1.0)

        _a_v = np.where(_valid, _a, 0.0)
        _r_v = np.where(_valid, np.broadcast_to(_r, _a.shape), 0.0)
        _a_mean = _a_v.sum(axis=1) / n_safe                        # (N,)
        _r_mean = _r_v.sum(axis=1) / n_safe                        # (N,)

        _da = np.where(_valid, _a - _a_mean[:, np.newaxis], 0.0)
        _dr = np.where(_valid, _r - _r_mean[:, np.newaxis], 0.0)
        _cov   = (_da * _dr).sum(axis=1) / n_safe
        _var_a = (_da ** 2).sum(axis=1) / n_safe
        _var_r = (_dr ** 2).sum(axis=1) / n_safe
        _denom = np.sqrt(_var_a * _var_r)
        corr_arr = np.where((n_valid > 1) & (_denom > 0), _cov / _denom, -np.inf)

        _diff2 = np.where(_valid, (_a - _r) ** 2, 0.0)
        rmse_arr = np.where(n_valid > 1, np.sqrt(_diff2.sum(axis=1) / n_safe), np.inf)

        return corr_arr, rmse_arr

    _win = 2 * gs_half_width + 1
    print(f"Grid search for best lon/lat offset (555 nm, {_win}x{_win} window)...")
    # The search starts with a relatively large span (e.g., ±0.015°) and iteratively narrows down to a finer span (e.g., ±0.0001°) around the best offset found in the previous iteration. 
    # The steps are designed to be denser near the center to efficiently find the optimal offset.
    # ------------------------------------------------------------------------------
    # _cos_lat = np.cos(np.deg2rad(np.nanmean(rsp_lat_all[:, 0])))  # equalize lon/lat steps in physical distance
    # _center_lon, _center_lat = 0.0, 0.0
    # _best_d_lon, _best_d_lat, _best_corr, _best_rmse = 0.0, 0.0, -np.inf, np.inf
    # _span = 0.015
    # _min_step = 0.0001
    # while _span >= _min_step:
    #     _pts_lon = _center_lon + np.array([-_span, -_span*2/3, -_span/3, 0.0, _span/3, _span*2/3, _span])
    #     _pts_lat = _center_lat + (_pts_lon - _center_lon) * _cos_lat
    #     _lon_grid, _lat_grid = np.meshgrid(_pts_lon, _pts_lat, indexing='ij')
    #     _corr_grid, _rmse_grid = _eval_corr(_lon_grid.ravel(), _lat_grid.ravel())
    #     _corr_grid = _corr_grid.reshape(len(_pts_lon), len(_pts_lat))
    #     _rmse_grid = _rmse_grid.reshape(len(_pts_lon), len(_pts_lat))
    #     _bi, _bj = np.unravel_index(np.argmax(_corr_grid), _corr_grid.shape)
    #     _best_d_lon, _best_d_lat = float(_pts_lon[_bi]), float(_pts_lat[_bj])
    #     _best_corr, _best_rmse = float(_corr_grid[_bi, _bj]), float(_rmse_grid[_bi, _bj])
    #     print(f"  span={_span:.4f} -> best d_lon={_best_d_lon:.4f}, d_lat={_best_d_lat:.4f}, corr={_best_corr:.4f}, rmse={_best_rmse:.4f}")
    #     _center_lon, _center_lat = _best_d_lon, _best_d_lat
    #     _span *= 0.5
    # print(f"Best offset: d_lon={_best_d_lon:.4f}, d_lat={_best_d_lat:.4f}, corr={_best_corr:.4f}, rmse={_best_rmse:.4f}")
    # ------------------------------------------------------------------------------
    
    # Optionally, a grid search around the original points found could be used instead of the above iterative refinement, but it would be more computationally expensive and less elegant.
    # ------------------------------------------------------------------------------
    _cos_lat = np.cos(np.deg2rad(np.nanmean(rsp_lat_all[:, 0])))  # equalize lon/lat steps in physical distance
    _pts_lon = np.arange(-0.015, 0.0151, 0.0001 )   # ±0.02° at 0.0005° steps
    _pts_lat = _pts_lon * _cos_lat
    # Evaluate all (d_lon, d_lat) grid points in one batch call
    _lon_grid, _lat_grid = np.meshgrid(_pts_lon, _pts_lat, indexing='ij')  # (n_lon, n_lat)
    _all_corr, _all_rmse = _eval_corr(_lon_grid.ravel(), _lat_grid.ravel())
    _grid_corr = _all_corr.reshape(len(_pts_lon), len(_pts_lat))
    _grid_rmse = _all_rmse.reshape(len(_pts_lon), len(_pts_lat))
    _best_flat = int(np.argmax(_grid_corr))
    _bi, _bj = np.unravel_index(_best_flat, _grid_corr.shape)
    _best_d_lon, _best_d_lat = float(_pts_lon[_bi]), float(_pts_lat[_bj])
    _best_corr, _best_rmse = float(_grid_corr[_bi, _bj]), float(_grid_rmse[_bi, _bj])
    print(f"Best offset: d_lon={_best_d_lon:.4f}, d_lat={_best_d_lat:.4f}, corr={_best_corr:.4f}, rmse={_best_rmse:.4f}")

    # 2-panel diagnostic: correlation and RMSE over the lon/lat offset grid
    plt.close('all')
    # _grid_corr[i, j] has axis-0=lon, axis-1=lat; transpose so rows=lat (y) and cols=lon (x)
    fig, (ax_c, ax_r) = plt.subplots(1, 2, figsize=(13, 5))
    _ext = [_pts_lon[0], _pts_lon[-1], _pts_lat[0], _pts_lat[-1]]  # [xmin, xmax, ymin, ymax]
    im_c = ax_c.imshow(_grid_corr.T, extent=_ext, origin='lower', aspect='auto', cmap='RdYlGn')
    ax_c.scatter(_best_d_lon, _best_d_lat, marker='*', color='k', s=150, zorder=5, label=f'best ({_best_d_lon:.4f}, {_best_d_lat:.4f})')
    fig.colorbar(im_c, ax=ax_c, label='Correlation')
    ax_c.set_xlabel('Δlon (°)')
    ax_c.set_ylabel('Δlat (°)')
    ax_c.set_title('Correlation vs offset (555 nm)')
    ax_c.legend(fontsize=9)

    im_r = ax_r.imshow(_grid_rmse.T, extent=_ext, origin='lower', aspect='auto', cmap='RdYlGn_r')
    ax_r.scatter(_best_d_lon, _best_d_lat, marker='*', color='k', s=150, zorder=5, label=f'best ({_best_d_lon:.4f}, {_best_d_lat:.4f})')
    fig.colorbar(im_r, ax=ax_r, label='RMSE (W m$^{-2}$ sr$^{-1}$ nm$^{-1}$)')
    ax_r.set_xlabel('Δlon (°)')
    ax_r.set_ylabel('Δlat (°)')
    ax_r.set_title('RMSE vs offset (555 nm)')
    ax_r.legend(fontsize=9)

    fig.suptitle(f'Grid-search offset landscape {date_s} for 3x3 AVIRIS pixels', fontsize=14)
    fig.tight_layout()
    savefig(fig, date_s, case_tag, 'offset_grid_search_corr_rmse')
    plt.close('all')
    # ------------------------------------------------------------------------------
    # --- end coarse-to-fine search ---

    (aviris_rad_all_1x1, aviris_rad_all_3x3, aviris_rad_all_5x5), _, near_lons, near_lats = \
        extract_aviris_spectrum(
            _aviris_rad_wvl, aviris_rad, aviris_rad_lon, aviris_rad_lat,
            rsp_lon_all[:, 0] + _best_d_lon, rsp_lat_all[:, 0] + _best_d_lat,
            half_width=[0, 1, 2],
        )
    # aviris_rad_all_*: (n_t, n_wvl)
    # summary map — all legs
    plt.close('all')
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_aviris_location_map(ax, aviris_rad_lon, aviris_rad_lat, rad_550,
                                lon_all, lat_all,
                                'AVIRIS L1B Radiance at 550 nm')
    fig.tight_layout()
    savefig(fig, date_s, case_tag, 'aviris_rad_550nm_all')
    
    
    rsp_wvl_list = [555, 863]
    rsp_wvl_arr = rsp_wvl_all  # save before the loop shadows it
    corr_coef_arr = np.full((len(rsp_wvl_list), 3), np.nan)  # (n_rsp_wvl, n_hw)
    for rsp_wvl in rsp_wvl_list:
        rsp_wvl_ind = np.argmin(np.abs(rsp_wvl_all - rsp_wvl))
        aviris_wvl_ind = np.argmin(np.abs(aviris_rad_wvl - rsp_wvl))
        plt.close('all')
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
        ax_list = [ax1, ax2, ax3]
        for i, aviris_spectrum, hw in zip(range(3),
                                          [aviris_rad_all_1x1, aviris_rad_all_3x3, aviris_rad_all_5x5], 
                                          [0, 1, 2]):
            print(f"AVIRIS spectrum with half_width={hw}: shape {aviris_spectrum.shape}")
        
            aviris_rad_ = aviris_spectrum[:, aviris_wvl_ind]
            ax = ax_list[i]
            ax.scatter(aviris_rad_, rsp_rad_all[:, rsp_wvl_ind], color='k', s=10, alpha=0.5)
            # calculate correlation coefficient
            valid_mask = ~np.isnan(aviris_rad_) & ~np.isnan(rsp_rad_all[:, rsp_wvl_ind])
            if np.sum(valid_mask) > 0:
                corr_coef = np.corrcoef(aviris_rad_[valid_mask], rsp_rad_all[valid_mask, rsp_wvl_ind])[0, 1]
                rmse = np.sqrt(np.nanmean((aviris_rad_[valid_mask] - rsp_rad_all[valid_mask, rsp_wvl_ind])**2))
                print(f"Correlation coefficient between AVIRIS and RSP at {rsp_wvl} nm ({2*hw+1}x{2*hw+1}): {corr_coef:.3f}, RMSE: {rmse:.4f}")
                ax.set_title(f'({2*hw+1}x{2*hw+1}, corr={corr_coef:.3f}, RMSE={rmse:.4f})', fontsize=16)
                corr_coef_arr[rsp_wvl_list.index(rsp_wvl), i] = corr_coef
            xy_min = np.nanmin([aviris_rad_, rsp_rad_all[:, rsp_wvl_ind]]) * 0.9
            xy_max = np.nanmax([aviris_rad_, rsp_rad_all[:, rsp_wvl_ind]]) * 1.1
            ax.plot([xy_min, xy_max], [xy_min, xy_max], 'r--', label='1:1 line')
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel('AVIRIS Radiance (W m$^{-2}$ sr$^{-1}$ nm$^{-1}$)', fontsize=14)
            ax.set_ylabel('RSP Radiance (W m$^{-2}$ sr$^{-1}$ nm$^{-1}$)', fontsize=14)
            ax.legend(fontsize=10)
            ax.tick_params(labelsize=12)
        fig.suptitle(f'RSP Nadir Radiance vs AVIRIS RDN at {rsp_wvl} nm, dlon={_best_d_lon:.4f}, dlat={_best_d_lat:.4f}',
                     fontsize=18, y=0.98)
        fig.tight_layout()
        savefig(fig, date_s, case_tag, f'aviris_rsp_rad_{rsp_wvl}nm_time_series')
    
    max_555_corr_idx = int(np.nanargmax(corr_coef_arr[0, :]))
    print(f"Best correlation at 555 nm with half_width={max_555_corr_idx*2+1}x{max_555_corr_idx*2+1}, corr={corr_coef_arr[0, max_555_corr_idx]:.3f}")
    aviris_rad_all = [aviris_rad_all_1x1, aviris_rad_all_3x3, aviris_rad_all_5x5][max_555_corr_idx]
    print("aviris_rad_all shape:", aviris_rad_all.shape)
    print("aviris_rad_wvl shape:", aviris_rad_wvl.shape)

    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4.5))
    ax1.plot(time_all, fdn_550_all, '-', color='b', label='550 nm downward')
    ax1.plot(time_all, fup_550_all, '-', color='r', label='550 nm upward')
    ax2.plot(time_all, fdn_1600_all, '-', color='b', label='1600 nm downward')
    ax2.plot(time_all, fup_1600_all, '-', color='r', label='1600 nm upward')
    ax1.set_xlabel('Time (UTC)', fontsize=14)
    ax1.set_ylabel('Flux (W/m$^2$/nm)', fontsize=14)
    ax1.set_title('SSFR 550 nm Flux', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.tick_params(labelsize=12)
    ax2.set_xlabel('Time (UTC)', fontsize=14)
    ax2.set_ylabel('Flux (W/m$^2$/nm)', fontsize=14)
    ax2.set_title('SSFR 1600 nm Flux', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.tick_params(labelsize=12)
    fig.suptitle(f'SSFR Flux Time Series {date_s}', fontsize=16, y=0.98)
    fig.tight_layout()
    savefig(fig, date_s, case_tag, 'ssfr_flux_time_series_all')
    

    aviris_rad_at_rsp_wvl = np.zeros((n_t, len(rsp_wvl_arr)))
    aviris_rad_at_ssrr_wvl = np.zeros((n_t, len(ssrr_rad_up_wvl)))
    ssrr_rad_at_rsp_wvl = np.zeros((n_t_ssrr, len(rsp_wvl_arr)))
    rsp_wvl_fwhm = np.array([27, 20, 20, 20, 20, 20, 60, 90, 130])
    
    
    ssrr_rad_interp = interp1d(ssrr_rad_up_wvl, ssrr_rad_up_all, 
                                kind='linear', bounds_error=False, fill_value=np.nan)
    aviris_rad_interp = interp1d(aviris_rad_wvl, aviris_rad_all,
                                    kind='linear', bounds_error=False, fill_value=np.nan)
    aviris_rad_at_ssrr_wvl = aviris_rad_interp(ssrr_rad_up_wvl)    

    for k in range(len(rsp_wvl_arr)):
        rsp_wvl_ = rsp_wvl_arr[k]
        rsl_wvl_fwhm_ = rsp_wvl_fwhm[k]
        rsp_wvl_range = np.arange(rsp_wvl_ - rsl_wvl_fwhm_*2, rsp_wvl_ + rsl_wvl_fwhm_*2+0.001, 1)
        print(f"RSP band {k}: center={rsp_wvl_} nm, FWHM={rsl_wvl_fwhm_} nm, range: {rsp_wvl_range[0]:.1f}-{rsp_wvl_range[-1]:.1f} nm")
        sigma = rsl_wvl_fwhm_ / (2 * np.sqrt(2 * np.log(2)))
        weights = np.exp(-0.5 * ((rsp_wvl_range - rsp_wvl_) / sigma) ** 2)
        weights /= np.sum(weights)
        # average ssrr rad over the fwhm range with gaussian
        ssrr_rad_at_rsp_wvl_range = ssrr_rad_interp(rsp_wvl_range)
        ssrr_rad_at_rsp_wvl[:, k] = np.nansum(ssrr_rad_at_rsp_wvl_range * weights[np.newaxis, :], axis=1)
        # average avris rad over the fwhm range with gaussian
        aviris_rad_at_rsp_wvl_range = aviris_rad_interp(rsp_wvl_range)
        aviris_rad_at_rsp_wvl[:, k] = np.nansum(aviris_rad_at_rsp_wvl_range * weights[np.newaxis, :], axis=1)
    
    # Interpolate SSRR (HSK time grid) and altitude to RSP's native time grid
    # so that rsp_rad_all (n_t=300) and the SSRR/alt arrays all share the same axis-0 length.
    ssrr_rad_at_rsp_wvl_interp = interp1d(time_all, ssrr_rad_at_rsp_wvl, axis=0, kind='nearest', bounds_error=False, fill_value=np.nan)
    ssrr_rad_at_rsp_wvl_interp_time = ssrr_rad_at_rsp_wvl_interp(rsp_time_all)
    alt_all_interp_at_rsp_time = interp1d(time_all, alt_all, kind='nearest', bounds_error=False, fill_value=np.nan)(rsp_time_all)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    scatter_kwargs = dict(marker='s', edgecolors='k', s=50, alpha=0.7)
    for j in range(len(rsp_wvl_arr)):
        wvl_repeat = np.repeat(rsp_wvl_arr[j], n_t)
        if j == 0:
            cs = ax.scatter(wvl_repeat, rsp_rad_all[:, j]/ssrr_rad_at_rsp_wvl_interp_time[:, j], c=alt_all_interp_at_rsp_time, **scatter_kwargs)        
        else:
            ax.scatter(wvl_repeat, rsp_rad_all[:, j]/ssrr_rad_at_rsp_wvl_interp_time[:, j], c=alt_all_interp_at_rsp_time, **scatter_kwargs)       
    cb = fig.colorbar(cs, ax=ax, label='Alt (km)')
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('Radiance Ratio', fontsize=14)
    ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.tick_params(labelsize=12)
    ax.set_ylim(0.8, 1.2)
    ax.set_xlim(350, 1000)
    ax.hlines(1.0, 350, 2200, color='gray', linestyle='--')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    fig.suptitle(f'RSP to SSRR Upward Radiance Ratio Comparison {date_s}', fontsize=13)
    fig.tight_layout()
    savefig(fig, date_s, case_tag, 'rsp_ssrr_rad_ratio_comparison_all')

    # AVIRIS vs RSP scatter plot — one subplot per RSP band, all 1Hz AVIRIS points
    n_rsp = len(rsp_wvl_arr)
    ncols = 3
    nrows = int(np.ceil(n_rsp / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).ravel()
    for k in range(n_rsp):
        ax = axes[k]
        x = aviris_rad_at_rsp_wvl[:, k]  # single value for the mean spectrum at this wavelength
        y = rsp_rad_all[:, k]
        alt_c = alt_all_interp_at_rsp_time
        sc = ax.scatter(x, y, c=alt_c, cmap='jet', s=20, edgecolors='none', alpha=0.5)
        fig.colorbar(sc, ax=ax, label='Alt (km)')
        vmin = np.nanmin([x, y])
        vmax = np.nanmax([x, y])
        ax.plot([vmin, vmax], [vmin, vmax], 'k--', linewidth=1.0, label='1:1')
        ax.set_xlabel('AVIRIS (W m$^{-2}$ sr$^{-1}$ nm$^{-1}$)', fontsize=10)
        ax.set_ylabel('RSP (W m$^{-2}$ sr$^{-1}$ nm$^{-1}$)', fontsize=10)
        ax.set_title(f'{rsp_wvl_arr[k]:.0f} nm', fontsize=11)
        ax.legend(fontsize=9)
        ax.tick_params(labelsize=9)
    for k in range(n_rsp, len(axes)):
        axes[k].set_visible(False)
    fig.suptitle(f'AVIRIS vs RSP Radiance {date_s}', fontsize=14)
    fig.tight_layout()
    savefig(fig, date_s, case_tag, 'aviris_rsp_rad_scatter')
    plt.close('all')
    
    # average aviris rad spectrum and RSP rad
    aviris_mean = np.nanmean(aviris_rad_all, axis=0)
    aviris_std  = np.nanstd(aviris_rad_all, axis=0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(aviris_rad_wvl, aviris_mean, 'm-', label='AVIRIS Mean Spectrum', alpha=0.7)
    ax.fill_between(aviris_rad_wvl, aviris_mean - aviris_std, aviris_mean + aviris_std, color='m', alpha=0.3)
    ax.errorbar(rsp_wvl_arr, np.nanmean(rsp_rad_all, axis=0), yerr=np.nanstd(rsp_rad_all, axis=0), fmt='o', color='c', label='RSP Nadir Radiance')
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('Radiance (W m$^{-2}$ sr$^{-1}$ nm$^{-1}$)', fontsize=14)
    _rsp_t0 = np.nanmin(rsp_time_all)
    _rsp_t1 = np.nanmax(rsp_time_all)
    _fmt_t = lambda h: str(datetime.timedelta(hours=float(h))).split('.')[0]
    ax.set_title(f'Average Spectrum Comparison {date_s}  RSP: {_fmt_t(_rsp_t0)}–{_fmt_t(_rsp_t1)} UTC', fontsize=12)
    ax.legend(fontsize=10)
    xmin, xmax = aviris_rad_wvl[0], aviris_rad_wvl[-1]
    ax.set_xlim(xmin, xmax)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    savefig(fig, date_s, case_tag, 'aviris_rsp_rad_spectrum_comparison')
    plt.close('all')


def ssfr_time_series_plot(data_hsk, data_ssfr, data_hsr1, tmhr_range, date_s, case_tag):
    t_hsk = np.array(data_hsk["tmhr"])    
    t_ssfr = data_ssfr['time']/3600.0  # convert to hours
    t_hsr1 = data_hsr1['time']/3600.0  # convert to hours
    
    pitch_ang = data_hsk['ang_pit']
    roll_ang = data_hsk['ang_rol']
    
    hsr_wvl = data_hsr1['wvl_dn_tot']
    hsr_ftot = data_hsr1['f_dn_tot']
    hsr_fdif = data_hsr1['f_dn_dif']
    hsr_dif_ratio = hsr_fdif/hsr_ftot
    hsr_550_ind = np.argmin(np.abs(hsr_wvl - 550))
    hsr_530_ind = np.argmin(np.abs(hsr_wvl - 530))
    hsr_570_ind = np.argmin(np.abs(hsr_wvl - 570))
    hsr1_diff_ratio = data_hsr1["f_dn_dif"]/data_hsr1["f_dn_tot"]
    hsr1_diff_ratio_530_570_mean = np.nanmean(hsr1_diff_ratio[:, hsr_530_ind:hsr_570_ind+1], axis=1)
        
    ssfr_zen_wvl = data_ssfr['wvl_dn']
    ssfr_fdn = data_ssfr['f_dn']
    ssfr_nad_wvl = data_ssfr['wvl_up']
    ssfr_fup = data_ssfr['f_up']
    
    t_ssfr_tmhr_mask = (t_ssfr >= tmhr_range[0]) & (t_ssfr <= tmhr_range[1])
    t_ssfr_tmhr = t_ssfr[t_ssfr_tmhr_mask]
    ssfr_fdn_tmhr = ssfr_fdn[t_ssfr_tmhr_mask]
    ssfr_fup_tmhr = ssfr_fup[t_ssfr_tmhr_mask]
    
    # apply data quality mask based on ssfr flags
    icing = (data_ssfr['flag'] & ssfr_flags.camera_icing) != 0
    pitch_roll_exceed = (data_ssfr['flag'] & ssfr_flags.pitch_roll_exceed_threshold) != 0
    alp_ang_pit_rol_issue = (data_ssfr['flag'] & ssfr_flags.alp_ang_pit_rol_issue) != 0
    problematic_issues = alp_ang_pit_rol_issue | pitch_roll_exceed | icing
    problematic_tmhr = problematic_issues[t_ssfr_tmhr_mask]
    ssfr_fdn_tmhr[problematic_tmhr] = np.nan
    ssfr_fup_tmhr[problematic_tmhr] = np.nan
    
    t_hsr1_tmhr_mask = (t_hsr1 >= tmhr_range[0]) & (t_hsr1 <= tmhr_range[1])
    t_hsr1_tmhr = t_hsr1[t_hsr1_tmhr_mask]
    hsr_dif_ratio = hsr_dif_ratio[t_hsr1_tmhr_mask]
    hsr1_diff_ratio_530_570_mean = hsr1_diff_ratio_530_570_mean[t_hsr1_tmhr_mask]
    
    ssfr_zen_550_ind = np.argmin(np.abs(ssfr_zen_wvl - 550))
    ssfr_nad_550_ind = np.argmin(np.abs(ssfr_nad_wvl - 550))
    
    fig, (ax10, ax20) = plt.subplots(2, 1, figsize=(16, 12))
    ax11 = ax10.twinx()
    l1 = ax10.plot(t_ssfr, ssfr_fdn[:, ssfr_zen_550_ind], '--', color='k', alpha=0.85)
    l2 = ax10.plot(t_ssfr, ssfr_fup[:, ssfr_nad_550_ind], '--', color='k', alpha=0.85)
    ax10.plot(t_ssfr_tmhr, ssfr_fdn_tmhr[:, ssfr_zen_550_ind], 'r-', label='SSFR Down 550nm', linewidth=3)
    ax10.plot(t_ssfr_tmhr, ssfr_fup_tmhr[:, ssfr_nad_550_ind], 'b-', label='SSFR Up 550nm', linewidth=3)
    l3 = ax11.plot(t_hsr1_tmhr, hsr_dif_ratio[:, hsr_550_ind], 'm-', label='HSR1 Diff Ratio 550nm')
    ax11.set_ylabel('HSR1 Diff Ratio 550nm', fontsize=14)
    
    ax20.plot(t_hsk, pitch_ang, 'g-', label='HSK Pitch')
    ax20.plot(t_hsk, roll_ang, 'b-', label='HSK Roll')
    lo, hi = tmhr_range[0], tmhr_range[-1]
    for ax in [ax10, ax20]:
        ax.fill_betweenx([0, 6], lo, hi, color='gray', alpha=0.3, transform=ax.get_xaxis_transform(),)# label=f'Leg {i+1}' if i==0 else None)
        ax.set_xlabel('Time (UTC)')
        ax.set_xlim(tmhr_range[0]*0.999, tmhr_range[1]*1.001)
    ll = l1 + l2 + l3
    labs = [l.get_label() for l in ll]
    ax10.legend(ll, labs, fontsize=10, )
    ax20.legend()
    ax10.set_ylim(0.4, 1.2)
    ax11.set_ylim(0.0, 0.65)
    ax10.set_ylabel('SSFR Flux (W/m$^2$/nm)')
    
    ax20.set_ylabel('HSK Pitch/Roll (deg)')
    ax20.set_ylim(-5, 5)
    fig.tight_layout()
    savefig(fig, date_s, case_tag, 'ssfr_pitch_roll_heading_550nm')



def flt_trk_data_collect(date=datetime.datetime(2024, 5, 31),
                     tmhr_range=[14.10, 14.27],
                     case_tag='default',
                     config: Optional[FlightConfig] = None,
                    ):

    log = logging.getLogger("data_collect")
    log.info("Starting processing for %s", date.strftime("%Y%m%d"))

    if config is None:
        raise ValueError("FlightConfig required")
    date_s = date.strftime("%Y%m%d")
    
    os.makedirs(f'fig/{date_s}', exist_ok=True)

    # 1) Load all instrument & satellite metadata
    data_hsk  = load_h5(config.hsk(date_s))
    data_ssfr = load_h5(config.ssfr(date_s))
    data_ssrr = load_h5(config.ssrr(date_s))
    data_hsr1 = load_h5(config.hsr1(date_s))
        
    log.info("ssfr filename: %s", config.ssfr(date_s))
    
    # plot ssfr time series for checking sable legs selection
    ssfr_time_series_plot(data_hsk, data_ssfr, data_hsr1, tmhr_range, date_s, case_tag)

    # Build leg masks
    t_hsk = np.array(data_hsk["tmhr"])
    lo, hi = tmhr_range[0], tmhr_range[-1]
    leg_mask = (t_hsk>=lo)&(t_hsk<=hi) 
    
    t_ssfr = data_ssfr['time']/3600.0  # convert to hours
    t_ssrr = data_ssrr['time']/3600.0  # convert to hours
 
    # Solar spectrum interpolation function
    flux_solar_interp = solar_interpolation_func(solar_flux_file='arcsix_ssfr_solar_flux_slit.dat', date=date)

    # check rsp l1b folder
    rsp_l1c_dir = f'{_fdir_general_}/rsp/ARCSIX-RSP-L1C_P3B_{date_s}_R01'
    rsp_l1c_files = np.array([], dtype=str)
    rsp_l1c_times = np.array([], dtype=float)
    if os.path.exists(rsp_l1c_dir):
        rsp_l1c_files = sorted(glob.glob(f'{rsp_l1c_dir}/ARCSIX-RSP-L1C_P3B_{date_s}*.h5'))
        print("rsp_l1c_files:", rsp_l1c_files)
        if len(rsp_l1c_files) == 0:
            print(f"No RSP L1B files found in {rsp_l1c_dir}")
            rsp_l1c_avail = False
        else:
            rsp_l1c_files = np.array(rsp_l1c_files)
            log.info(f"Found {len(rsp_l1c_files)} RSP L1C files in {rsp_l1c_dir}")
            # ARCSIX-RSP-L1B_P3B_20240605111213_R01.h5
            print([os.path.basename(f).split('_')[2] for f in rsp_l1c_files])
            rsp_l1c_times = np.array([int(os.path.basename(f).split('_')[2][8:10]) + int(os.path.basename(f).split('_')[2][10:12])/60.0 + int(os.path.basename(f).split('_')[2][12:14])/3600.0 for f in rsp_l1c_files])
            rsp_l1c_avail = True
    else:
        print(f"RSP L1B directory {rsp_l1c_dir} does not exist")
        rsp_l1c_avail = False
    
    if not rsp_l1c_avail:
        log.warning("RSP L1C data not available for %s, skipping RSP-related processing", date_s)
        raise FileNotFoundError(f"RSP L1C data not available for {date_s}")

    # read satellite granule
    #/----------------------------------------------------------------------------\#
    fdir_cld_obs_info = f'{_fdir_general_}/flt_cld_obs_info'
    os.makedirs(fdir_cld_obs_info, exist_ok=True)

    # find index arrays in one go
    times_leg = t_hsk[leg_mask]
    print(f" time range {times_leg.min()}-{times_leg.max()}h")
    
    sel_ssfr, sel_ssrr = (
        nearest_indices(t_hsk, leg_mask, arr)
        for arr in (t_ssfr, t_ssrr)
    )
    
    # choose the rsc l1b file for this leg
    time_leg_start = times_leg.min()
    time_leg_end = times_leg.max()
    rsp_l1c_sel = np.zeros(rsp_l1c_times.shape, dtype=bool)
    rsp_file_start = np.where(rsp_l1c_times==rsp_l1c_times[((rsp_l1c_times - time_leg_start)<0)][-1])[0][0]
    try:
        rsp_file_end = np.where(rsp_l1c_times==rsp_l1c_times[((rsp_l1c_times - time_leg_end)>0)][0])[0][0]
        rsp_l1c_sel[rsp_file_start:rsp_file_end] = True
    except IndexError:
        rsp_l1c_sel[rsp_file_start:] = True
    if np.sum(rsp_l1c_sel) == 0:
        log.warning("No RSP L1C files found for time range %.3f-%.3fh", time_leg_start, time_leg_end)
        rsp_l1c_files_leg = np.array([])
    else:
        rsp_l1c_files_leg = rsp_l1c_files[rsp_l1c_sel]
    
    rsp_time_all = []
    rsp_rad_all = []
    rsp_rad_norm_all = []
    rsp_lon_all = []
    rsp_lat_all = []
    rsp_mu0_all = []
    rsp_sd_all = []
    
    rsp_rad_1880_all = []
    rsp_rad_norm_1880_all = []
    rsp_lon_1880_all = []
    rsp_lat_1880_all = []
    rsp_mu0_1880_all = []
    
    
    band_1880_ind = 7  # RSP 1880 nm band index (updated from file data in loop)
    for rsp_file_name in rsp_l1c_files_leg:
        log.info(f"Reading RSP L1C file: {rsp_file_name}")
        data_rsp = load_h5(rsp_file_name)
        t_rsp = data_rsp['Platform/Fraction_of_Day']*24.0  # in hours, (dim_Scans)
        rsp_ground_lat_1880 = data_rsp['Geometry/Ground_Latitude'][1, :, :]  # (dim_Scans, dim_Scene_Sectors)
        rsp_ground_lat = data_rsp['Geometry/Ground_Latitude'][0, :, :]  # (dim_Scans, dim_Scene_Sectors)
        rsp_ground_lon_1880 = data_rsp['Geometry/Ground_Longitude'][1, :, :]  # (dim_Scans, dim_Scene_Sectors)
        rsp_ground_lon = data_rsp['Geometry/Ground_Longitude'][0, :, :]  # (dim_Scans, dim_Scene_Sectors)
        rsp_sza_1880 = data_rsp['Geometry/Solar_Zenith'][1, :, :]  # in degree, (dim_Scans, dim_Scene_Sectors), for the 1880nm band, index = 7
        rsp_sza = data_rsp['Geometry/Solar_Zenith'][0, :, :]  # in degree, (dim_Scans, dim_Scene_Sectors)
        rsp_vza_1880 = data_rsp['Geometry/Viewing_Zenith'][1, :, :]  # in degrees, (dim_Scans, dim_Scene_Sectors)
        rsp_vza = data_rsp['Geometry/Viewing_Zenith'][0, :, :]  # in degrees, (dim_Scans, dim_Scene_Sectors)
        rsp_nadir_ind_1880 = data_rsp['Geometry/Nadir_Index'][1, :]
        rsp_nadir_ind = data_rsp['Geometry/Nadir_Index'][0, :]
        rsp_sd = data_rsp['Platform/Solar_Distance']  # (dim_Scans), in AU
        
        intensity_1 = data_rsp['Data/Intensity_1']  # (dim_Scans, dim_Scene_Sectors, bands)
        intensity_2 = data_rsp['Data/Intensity_2']  # (dim_Scans, dim_Scene_Sectors, bands)
        intensity_1[intensity_1 < 0] = np.nan
        intensity_2[intensity_2 < 0] = np.nan
        rsp_wvl = data_rsp['Data/Wavelength']  # in nm, (bands,)
        rsp_solar_const = data_rsp['Calibration/Solar_Constant']  # in W/m^2/nm, (bands,)
        
        # sel_rsp = nearest_indices(t_hsk, leg_mask, t_rsp)
        sel_rsp = np.logical_and(t_rsp >= time_leg_start, t_rsp <= time_leg_end)
        rsp_time_sel = t_rsp[sel_rsp]
        
        # calculate all wavelengths first
        rsp_int_1_sel = intensity_1[sel_rsp, rsp_nadir_ind[sel_rsp], :]  # (time, bands)
        rsp_int_2_sel = intensity_2[sel_rsp, rsp_nadir_ind[sel_rsp], :]  # (time, bands)
        rsp_lon_sel = rsp_ground_lon[sel_rsp, rsp_nadir_ind[sel_rsp]]
        rsp_lat_sel = rsp_ground_lat[sel_rsp, rsp_nadir_ind[sel_rsp]]
        rsp_sza_sel = rsp_sza[sel_rsp, rsp_nadir_ind[sel_rsp]]
        rsp_sd_sel = rsp_sd[sel_rsp]
        
        rsp_rad_norm = (rsp_int_1_sel + rsp_int_2_sel) / 2  # (time, bands), in counts
        # rsp_rad = rsp_rad_norm * rsp_solar_const[np.newaxis, :] * np.cos(np.deg2rad(rsp_sza_sel))[:, np.newaxis] / rsp_sd_sel[:, np.newaxis]**2 / np.pi
        rsp_rad = rsp_rad_norm * rsp_solar_const[np.newaxis, :] / np.pi
        rsp_rad[rsp_rad < 0] = np.nan
        
        rsp_time_all.extend(rsp_time_sel)
        rsp_rad_all.extend(rsp_rad)
        rsp_rad_norm_all.extend(rsp_rad_norm)
        rsp_lon_all.extend(rsp_lon_sel)
        rsp_lat_all.extend(rsp_lat_sel)
        rsp_mu0_all.extend(np.cos(np.deg2rad(rsp_sza_sel)))
        rsp_sd_all.extend(rsp_sd_sel)
        
        # calculate the 1880nm band separately
        band_1880_ind = np.argmin(np.abs(rsp_wvl - 1880))
        rsp_int_1_sel_1880 = intensity_1[sel_rsp, rsp_nadir_ind_1880[sel_rsp], :]  # (time, bands)
        rsp_int_2_sel_1880 = intensity_2[sel_rsp, rsp_nadir_ind_1880[sel_rsp], :]  # (time, bands)
        rsp_lon_sel_1880 = rsp_ground_lon_1880[sel_rsp, rsp_nadir_ind_1880[sel_rsp]]
        rsp_lat_sel_1880 = rsp_ground_lat_1880[sel_rsp, rsp_nadir_ind_1880[sel_rsp]]
        rsp_sza_sel_1880 = rsp_sza_1880[sel_rsp, rsp_nadir_ind_1880[sel_rsp]]
        
        rsp_rad_norm_1880 = (rsp_int_1_sel_1880 + rsp_int_2_sel_1880) / 2  # (time, bands), in counts
        # rsp_rad_1880 = rsp_rad_norm_1880 * rsp_solar_const[band_1880_ind] * np.cos(np.deg2rad(rsp_sza_sel_1880))[:, np.newaxis] / rsp_sd_sel_1880[:, np.newaxis]**2 / np.pi
        rsp_rad_1880 = rsp_rad_norm_1880 * rsp_solar_const[band_1880_ind] / np.pi
        rsp_rad_1880[rsp_rad_1880 < 0] = np.nan
        
        rsp_rad_1880_all.extend(rsp_rad_1880)
        rsp_rad_norm_1880_all.extend(rsp_rad_norm_1880)
        rsp_lon_1880_all.extend(rsp_lon_sel_1880)
        rsp_lat_1880_all.extend(rsp_lat_sel_1880)
        rsp_mu0_1880_all.extend(np.cos(np.deg2rad(rsp_sza_sel_1880)))
            

    rsp_rad_all = np.array(rsp_rad_all)
    rsp_rad_norm_all = np.array(rsp_rad_norm_all)
    rsp_lon_all = np.array(rsp_lon_all)
    rsp_lat_all = np.array(rsp_lat_all)
    rsp_mu0_all = np.array(rsp_mu0_all)
    rsp_sd_all = np.array(rsp_sd_all)
    
    rsp_rad_1880_all = np.array(rsp_rad_1880_all)
    rsp_rad_norm_1880_all = np.array(rsp_rad_norm_1880_all)
    rsp_lon_1880_all = np.array(rsp_lon_1880_all)
    rsp_lat_1880_all = np.array(rsp_lat_1880_all)
    rsp_mu0_1880_all = np.array(rsp_mu0_1880_all)
    
    rsp_rad_all[:, band_1880_ind] = rsp_rad_all[:, band_1880_ind]
    rsp_rad_norm_all[:, band_1880_ind] = rsp_rad_norm_all[:, band_1880_ind]

    rsp_lon_all = np.repeat(rsp_lon_all[:, np.newaxis], len(rsp_wvl), axis=1)
    rsp_lat_all = np.repeat(rsp_lat_all[:, np.newaxis], len(rsp_wvl), axis=1)
    rsp_mu0_all = np.repeat(rsp_mu0_all[:, np.newaxis], len(rsp_wvl), axis=1)
    rsp_lon_all[:, band_1880_ind] = rsp_lon_1880_all
    rsp_lat_all[:, band_1880_ind] = rsp_lat_1880_all
    rsp_mu0_all[:, band_1880_ind] = rsp_mu0_1880_all

    rsp_rad_all /= 1000.0  # convert to W/m^2/sr/nm

    log.info("rsp_rad_all shape: %s, min/max: %.4f / %.4f",
                rsp_rad_all.shape, np.nanmin(rsp_rad_all), np.nanmax(rsp_rad_all))

    # assemble a small dict for this leg
    leg = {
        "time":    times_leg,
        "alt":     data_hsk["alt"][leg_mask] / 1000.0,
        "heading": data_hsk["ang_hed"][leg_mask],\
        "lon":     data_hsk["lon"][leg_mask],
        "lat":     data_hsk["lat"][leg_mask],
        "sza":     data_hsk["sza"][leg_mask],
        "saa":     data_hsk["saa"][leg_mask],
        "p3_alt":  data_hsk["alt"][leg_mask] / 1000.0,
        
        "rsp_time": np.array(rsp_time_all),
        "rsp_lon": np.array(rsp_lon_all),
        "rsp_lat": np.array(rsp_lat_all),
        "rsp_rad": np.array(rsp_rad_all),
        "rsp_rad_norm": np.array(rsp_rad_norm_all),
        "rsp_wvl": rsp_wvl,
        "rsp_mu0": np.array(rsp_mu0_all),
        "rsp_sd":  np.array(rsp_sd_all),   
    }
    
    sza_hsk = data_hsk['sza'][leg_mask]

    ssfr_zen_flux = data_ssfr['f_dn'][sel_ssfr, :]
    ssfr_nad_flux = data_ssfr['f_up'][sel_ssfr, :]
    ssfr_zen_toa = flux_solar_interp(data_ssfr['wvl_dn']) * np.cos(np.deg2rad(sza_hsk))[:, np.newaxis]  # W/m^2/nm
    ssfr_zen_wvl = data_ssfr['wvl_dn']
    ssfr_nad_wvl = data_ssfr['wvl_up']
    
    # interpolate ssfr nadir flux to zenith wavelength grid for easier comparison with ssfr zenith and ssrr
    # nadir and zenith channels have different native wavelength grids; regrid nadir onto zenith grid
    ssfr_nad_flux_interp = interp1d(
        ssfr_nad_wvl, ssfr_nad_flux, axis=1, bounds_error=False, fill_value='extrapolate'
    )(ssfr_zen_wvl)
    
    # apply data quality mask based on ssfr flags
    icing = (data_ssfr['flag'] & ssfr_flags.camera_icing) != 0
    pitch_roll_exceed = (data_ssfr['flag'] & ssfr_flags.pitch_roll_exceed_threshold) != 0
    alp_ang_pit_rol_issue = (data_ssfr['flag'] & ssfr_flags.alp_ang_pit_rol_issue) != 0
    problematic_issues = alp_ang_pit_rol_issue | pitch_roll_exceed | icing
    problematic_tmhr = problematic_issues[sel_ssfr]
    ssfr_zen_flux[problematic_tmhr] = np.nan
    ssfr_nad_flux_interp[problematic_tmhr] = np.nan
    
    leg['ssfr_zen'] = ssfr_zen_flux
    leg['ssfr_nad'] = ssfr_nad_flux_interp
    leg['ssfr_zen_wvl'] = ssfr_zen_wvl
    leg['ssfr_nad_wvl'] = ssfr_zen_wvl
    leg['ssfr_toa'] = ssfr_zen_toa
    
    # ssrr
    # interpolate ssrr zenith radiance to nadir wavelength grid
    f_zen_rad_interp = interp1d(data_ssrr["wvl_dn"], data_ssrr["i_dn"][sel_ssrr, :], axis=1, bounds_error=False, fill_value=np.nan)
    ssrr_rad_zen_i = f_zen_rad_interp(ssfr_zen_wvl)
    f_nad_rad_interp = interp1d(data_ssrr["wvl_up"], data_ssrr["i_up"][sel_ssrr, :], axis=1, bounds_error=False, fill_value=np.nan)
    ssrr_rad_nad_i = f_nad_rad_interp(ssfr_zen_wvl)
    leg['ssrr_zen_rad'] = ssrr_rad_zen_i
    leg['ssrr_nad_rad'] = ssrr_rad_nad_i
    leg['ssrr_zen_wvl'] = ssfr_zen_wvl
    leg['ssrr_nad_wvl'] = ssfr_zen_wvl

    # save the cloud observation information to a pickle file
    time_start, time_end = tmhr_range[0], tmhr_range[-1]
    fname_pkl = '%s/%s_cld_obs_info_%s_%s_%s_time_%.3f-%.3f_atm_corr.pkl' % (fdir_cld_obs_info, _mission_.lower(), _platform_.lower(), date_s, case_tag, time_start, time_end)            
    with open(fname_pkl, 'wb') as f:
        pickle.dump(leg, f, protocol=pickle.HIGHEST_PROTOCOL)

    del leg  # free memory
    del sel_ssfr, sel_ssrr
    if rsp_l1c_avail:
        del rsp_time_all, rsp_rad_all, rsp_lon_all, rsp_lat_all
        del rsp_mu0_all, rsp_sd_all
        del data_rsp
        del t_rsp, intensity_1, intensity_2, rsp_solar_const
        del rsp_ground_lon, rsp_ground_lat, rsp_sza, rsp_vza
    gc.collect()
    return fname_pkl
        
if __name__ == '__main__':

    dir_fig = './fig'
    os.makedirs(dir_fig, exist_ok=True)
    
    config = FlightConfig(mission='ARCSIX',
                            platform='P3B',
                            data_root=_fdir_data_,
                            root_mac=_fdir_general_,
                            root_linux='/pl/active/vikas-arcsix/yuch8913/arcsix/data',)

    # flt_trk_data_collect(
    #                  date=datetime.datetime(2024, 6, 6),
    #                  tmhr_range=[16.338, 16.411], # 450m
    #                  case_tag='rsp_comparison',
    #                  config=config,
    #                 )

    atm_corr_plot(date=datetime.datetime(2024, 6, 6),
                    tmhr_range=[16.338 , 16.411], # 450m
                    case_tag='rsp_comparison',
                    config=config,
                    aviris_id='ang20240606t161246_002',
                    )