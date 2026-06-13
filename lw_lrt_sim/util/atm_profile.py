"""
Standalone atmospheric profile preparation for LW simulations.

This module replicates the functionality of prepare_atmospheric_profile from
lrt_sim/util/arcsix_atm.py without importing er3t. Key differences:
  - MODIS MOD-07 file is provided directly (no download step)
  - No dropsonde, in-situ, or MARLi water-vapour inputs
  - Only MODIS + climatology data are used

Acknowledgement
---------------
Logic in this module is adapted from EaR³T
(Education and Research 3D Radiative Transfer Toolbox):

  Authors : Vikas Nataraja, Yu-Wen Chen, Ken Hirata, Hong Chen, Sebastian Schmidt
  GitHub  : https://github.com/hong-chen/er3t
  Paper   : Chen et al. (2023), AMT, doi:10.5194/amt-16-1971-2023
  License : GNU GPLv3

Original sources:
  er3t.pre.atm.create_modis_dropsonde_atm       → er3t/pre/atm/
  er3t.pre.atm.modis_dropsonde_arcsix_atmmod    → er3t/pre/atm/
"""

import os
import sys
import copy
import bisect
import datetime

import h5py
import numpy as np
import netCDF4 as nc
from scipy.interpolate import interp1d
from scipy import interpolate


# ─── HDF4 helper ──────────────────────────────────────────────────────────────

def _get_data_h4(hdf_dset, init_dtype=None, replace_fill_value=np.nan):
    """Read an HDF4 dataset, apply scale/offset, and replace fill values."""
    attrs = hdf_dset.attributes()
    data  = hdf_dset[:]
    if init_dtype is not None:
        data = np.array(data, dtype=init_dtype)

    if '_FillValue' in attrs and replace_fill_value is not None:
        if np.isnan(replace_fill_value):
            _FillValue = np.array(attrs['_FillValue'], dtype='float64')
            data = data.astype('float64')
        else:
            _FillValue = np.array(attrs['_FillValue'], dtype=data.dtype)
        data[data == _FillValue] = replace_fill_value

    if 'add_offset' in attrs:
        data = data - attrs['add_offset']
    if 'scale_factor' in attrs:
        data = data * attrs['scale_factor']

    return data


# ─── MODIS MOD-07 reader ──────────────────────────────────────────────────────

def _read_modis_07(fname, extent):
    """
    Read a single MOD07/MYD07 HDF4 file and return spatially filtered data.

    Parameters
    ----------
    fname : str
        Path to a MOD07/MYD07 HDF4 file.
    extent : list
        [lon_west, lon_east, lat_south, lat_north]

    Returns
    -------
    dict with keys matching the er3t modis_07.data dict:
        p_level, cld_mask, T_level_retrieved, h_level_retrieved,
        dewT_level_retrieved, wvmx_level_retrieved, h_sfc, p_sfc,
        t_skin, sza, vza
    """
    try:
        from pyhdf.SD import SD, SDC
    except ImportError:
        raise ImportError("pyhdf is required to read MODIS HDF4 files. Install with: pip install pyhdf")

    f = SD(fname, SDC.READ)

    lon = f.select('Longitude')[:]
    lat = f.select('Latitude')[:]

    lon_range = [extent[0] - 0.01, extent[1] + 0.01]
    lat_range = [extent[2] - 0.01, extent[3] + 0.01]
    logic = (lon >= lon_range[0]) & (lon <= lon_range[1]) & \
            (lat >= lat_range[0]) & (lat <= lat_range[1])

    p_level = np.array([5, 10, 20, 30, 50, 70, 100, 150, 200, 250,
                         300, 400, 500, 620, 700, 780, 850, 920, 950, 1000],
                        dtype='float64')

    # Cloud mask (fov_qa_cat: 0=cloudy, 1=uncertain, 2=prob clear, 3=confident clear)
    cm0_data = _get_data_h4(f.select('Cloud_Mask'), replace_fill_value=None)
    cm = np.array(cm0_data[logic], dtype='uint8').reshape(-1, 1)
    cm_bits = np.unpackbits(cm, bitorder='big', axis=1)
    fov_qa_cat = 2 * cm_bits[:, 5] + 1 * cm_bits[:, 6]

    T_lev   = np.array(_get_data_h4(f.select('Retrieved_Temperature_Profile'))[:, logic])
    h_lev   = np.array(_get_data_h4(f.select('Retrieved_Height_Profile'))[:, logic])
    dewT    = np.array(_get_data_h4(f.select('Retrieved_Moisture_Profile'))[:, logic])
    wvmx    = np.array(_get_data_h4(f.select('Retrieved_WV_Mixing_Ratio_Profile'))[:, logic])
    h_sfc   = np.array(_get_data_h4(f.select('Surface_Elevation'))[logic])
    p_sfc   = np.array(_get_data_h4(f.select('Surface_Pressure'))[logic])
    t_skin  = np.array(_get_data_h4(f.select('Skin_Temperature'))[logic])
    sza     = np.array(_get_data_h4(f.select('Solar_Zenith'))[logic])
    vza     = np.array(_get_data_h4(f.select('Sensor_Zenith'))[logic])

    for arr in [T_lev, h_lev, dewT, wvmx, h_sfc, p_sfc]:
        arr[arr < 0] = np.nan

    f.end()

    return {
        'p_level':              p_level,
        'cld_mask':             fov_qa_cat,
        'T_level_retrieved':    T_lev,
        'h_level_retrieved':    h_lev,
        'dewT_level_retrieved': dewT,
        'wvmx_level_retrieved': wvmx,
        'h_sfc':                h_sfc,
        'p_sfc':                p_sfc,
        't_skin':               t_skin,
        'sza':                  sza,
        'vza':                  vza,
    }


# ─── ZPT file creation (MODIS only, no dropsonde) ─────────────────────────────

def _create_modis_atm_zpt(fname_mod07, extent, output_dir, output,
                           o2mix=0.20935, levels=None,
                           sfc_T_set=None, sfc_h_to_zero=True):
    """
    Build a ZPT HDF5 file from a MODIS MOD-07 file.

    Parameters
    ----------
    fname_mod07 : str
        Path to the MOD07/MYD07 HDF4 file.
    extent : list
        [lon_west, lon_east, lat_south, lat_north]
    output_dir : str
        Directory for the output HDF5 file.
    output : str
        Output HDF5 filename.
    o2mix : float
        O2 volume mixing ratio (default 0.20935).
    levels : array-like or None
        Simulation altitude levels in km. If None, defaults are used.
    sfc_T_set : float or None
        Override surface temperature (K). None keeps the MODIS value.
    sfc_h_to_zero : bool
        Force surface height to 0.

    Returns
    -------
    str : 'success' or 'error'
    """
    EPSILON = 0.622

    # Normalise extent to [lon_min, lon_max, lat_min, lat_max]
    ext = [min(extent[0], extent[1]), max(extent[0], extent[1]),
           min(extent[2], extent[3]), max(extent[2], extent[3])]

    # Expand extent slightly until valid MODIS data is found
    sfc_p = [np.nan]
    try_time = 0
    while try_time < 15 and np.isnan(np.nanmean(sfc_p)):
        bw = 0.1 * try_time
        ext_try = [ext[0] - bw, ext[1] + bw, ext[2] - bw, ext[3] + bw]
        try_time += 1
        mod07 = _read_modis_07(fname_mod07, ext_try)
        sfc_p = mod07['p_sfc']

    if np.isnan(np.nanmean(sfc_p)):
        print("[Error] create_modis_atm_zpt: No valid MODIS surface pressure found.")
        return 'error'

    pprf_l_single = mod07['p_level']                          # (20,)  hPa
    hprf_l        = mod07['h_level_retrieved']                # (20, N) m
    tprf_l        = mod07['T_level_retrieved']                # (20, N) K
    mwvmxprf_l    = mod07['wvmx_level_retrieved']             # (20, N) g/kg
    h_sfc         = mod07['h_sfc']                            # (N,)   m

    # Broadcast pressure to (20, N)
    pprf_l = np.repeat(pprf_l_single, mwvmxprf_l.shape[1]).reshape(mwvmxprf_l.shape)

    r      = mwvmxprf_l / 1000.0                              # g/kg -> kg/kg
    eprf_l = pprf_l * r / (EPSILON + r)
    h2o_vmr = eprf_l / (pprf_l - eprf_l)

    sfc_p_mean    = np.nanmean(sfc_p)
    sfc_h_mean    = np.nanmean(h_sfc)
    pprf_lev_mean = np.nanmean(pprf_l,   axis=1)
    tprf_lev_mean = np.nanmean(tprf_l,   axis=1)
    hprf_lev_mean = np.nanmean(hprf_l,   axis=1) / 1000.0    # m -> km
    h2o_vmr_mean  = np.nanmean(h2o_vmr,  axis=1)

    # Trim levels with NaN height at the top
    while np.isnan(hprf_lev_mean[-1]):
        pprf_lev_mean = pprf_lev_mean[:-1]
        tprf_lev_mean = tprf_lev_mean[:-1]
        hprf_lev_mean = hprf_lev_mean[:-1]
        h2o_vmr_mean  = h2o_vmr_mean[:-1]

    if sfc_h_to_zero:
        sfc_h_mean = 0.0

    # Extrapolate to surface
    f_temp    = interp1d(pprf_lev_mean[:-1], tprf_lev_mean[:-1],  fill_value='extrapolate')
    f_h2o_vmr = interp1d(pprf_lev_mean[:-1], h2o_vmr_mean[:-1],   fill_value='extrapolate')
    pprf_lev_mean[-1]  = sfc_p_mean
    tprf_lev_mean[-1]  = f_temp(sfc_p_mean)
    hprf_lev_mean[-1]  = sfc_h_mean
    h2o_vmr_mean[-1]   = f_h2o_vmr(sfc_p_mean)

    if sfc_T_set is not None:
        tprf_lev_mean[-1] = sfc_T_set

    # Simulation levels
    if levels is not None:
        sim_levels = np.array(levels)
    else:
        sim_levels = np.concatenate((
            np.linspace(sfc_h_mean, 4.0, 11),
            np.arange(5.0, 10.1, 1.0),
            np.array([12.5, 15, 17.5, 20., 25., 30., 40.])
        ))

    output_path = os.path.join(output_dir, output)
    print(f'Saving ZPT to {output_path}')
    with h5py.File(output_path, 'w') as f:
        # Variables read by _build_atm_profile
        f.create_dataset('level_sim',    data=sim_levels)
        f.create_dataset('h_lev',        data=hprf_lev_mean)
        f.create_dataset('p_lev',        data=pprf_lev_mean)
        f.create_dataset('t_lev',        data=tprf_lev_mean)
        f.create_dataset('h2o_vmr',      data=h2o_vmr_mean)
        f.create_dataset('o2_mix',       data=o2mix)

    return 'success'


# ─── Barometric pressure interpolation ────────────────────────────────────────

def _atm_interp_pressure(pressure, altitude, temperature,
                          altitude_to_interp, temperature_to_interp):
    """
    Interpolate pressure using the Barometric formula.

    Parameters are identical to er3t's atm_interp_pressure.
    """
    indices = np.argsort(altitude)
    h = np.float64(altitude[indices])
    p = np.float64(pressure[indices])
    t = np.float64(temperature[indices])

    indices = np.argsort(altitude_to_interp)
    hn = np.float64(altitude_to_interp[indices])
    tn = np.float64(temperature_to_interp[indices])

    n  = p.size - 1
    a  = 0.5 * (t[1:] + t[:-1]) / (h[:-1] - h[1:]) * np.log(p[1:] / p[:-1])
    z  = 0.5 * (h[1:] + h[:-1])

    z0  = z.min();  z1  = z.max()
    hn0 = hn.min(); hn1 = hn.max()

    if hn0 < z0:
        a = np.hstack((a[0], a))
        z = np.hstack((hn0,  z))
        if z0 - hn0 > 2.0:
            print('Warning [_atm_interp_pressure]: Standard atmosphere not sufficient (lower boundary).')

    if hn1 > z1:
        a = np.hstack((a, z[n - 1]))
        z = np.hstack((z, hn1))
        if hn1 - z1 > 10.0:
            print('Warning [_atm_interp_pressure]: Standard atmosphere not sufficient (upper boundary).')

    an = np.interp(hn, z, a)
    pn = np.zeros_like(hn)

    if hn.size == 1:
        hi = np.argmin(np.abs(hn - h))
        pn = p[hi] * np.exp(-an * (hn - h[hi]) / tn)
        return pn

    for i in range(pn.size):
        hi    = np.argmin(np.abs(hn[i] - h))
        pn[i] = p[hi] * np.exp(-an[i] * (hn[i] - h[hi]) / tn[i])

    dp = pn[:-1] - pn[1:]
    pl = 0.5 * (pn[1:] + pn[:-1])
    zl = 0.5 * (hn[1:] + hn[:-1])

    for i in range(n - 2):
        indices = (zl >= h[i]) & (zl < h[i + 1])
        ind = np.where(indices)[0]
        ni  = indices.sum()
        if ni >= 2:
            dpm = dp[ind].sum()
            i0, i1 = ind.min(), ind.max()
            x1, x2 = pl[i0], pl[i1]
            y1, y2 = dp[i0], dp[i1]
            bb = (y2 - y1) / (x2 - x1)
            aa = y1 - bb * x1
            rescale = dpm / (aa + bb * pl[indices]).sum()
            if np.abs(rescale - 1.0) <= 0.1:
                dp[indices] = rescale * (aa + bb * pl[indices])
            else:
                print(f'Warning [_atm_interp_pressure]: pressure smoothing failed at '
                      f'{h[i]:.3f}...{h[i+1]:.3f} km (rescale={rescale:.4f})')

    for i in range(dp.size):
        pn[i + 1] = pn[i] - dp[i]

    return pn


# ─── CH4 default altitude profile ─────────────────────────────────────────────

def _atm_interp_ch4(altitude_inp):
    """Return CH4 volume mixing ratio for given altitudes (km)."""
    ch4h = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 3.0, 4.0,
                     5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                     15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                     25.0, 27.0, 29.0, 31.0, 33.0, 35.0, 37.0, 40.0])
    ch4m = np.array([1.70000e-06, 1.70000e-06, 1.70000e-06, 1.70000e-06,
                     1.70000e-06, 1.70000e-06, 1.70000e-06, 1.70000e-06,
                     1.70000e-06, 1.70000e-06, 1.70000e-06, 1.70000e-06,
                     1.69900e-06, 1.69700e-06, 1.69300e-06, 1.68500e-06,
                     1.67485e-06, 1.66200e-06, 1.64753e-06, 1.62915e-06,
                     1.60500e-06, 1.58531e-06, 1.55875e-06, 1.52100e-06,
                     1.48145e-06, 1.42400e-06, 1.38858e-06, 1.34258e-06,
                     1.28041e-06, 1.19173e-06, 1.05500e-06, 1.02223e-06,
                     9.63919e-07, 9.04935e-07, 8.82387e-07, 8.48513e-07,
                     7.91919e-07, 0.0])
    return np.interp(altitude_inp, ch4h, ch4m)


# ─── Climatology readers ───────────────────────────────────────────────────────

def _read_co2_clim(fname_co2_clim, date, extent):
    """Return (pressure_hPa, co2_vmr) from CAMS CO2 climatology NetCDF."""
    with nc.Dataset(fname_co2_clim, mode='r') as f:
        month        = date.month
        co2_lon      = f.variables['longitude'][:]
        co2_lat      = f.variables['latitude'][:]
        co2_pressure = f.variables['pressure'][:]
        co2_clim     = f.variables[f'CO2_{month:02d}'][:].T  # vmr

        num_levels  = len(co2_pressure)
        co2_loc     = np.zeros(num_levels)
        lon_mid     = np.mean(extent[:2])
        lat_mid     = np.mean(extent[2:])
        for i in range(num_levels):
            f_co2      = interpolate.interp2d(co2_lon, co2_lat, co2_clim[:, :, i].T, kind='linear')
            co2_loc[i] = f_co2(lon_mid, lat_mid)

        mask = co2_loc < 0
        if not mask.all():
            co2_loc[mask] = co2_loc[~mask].max()

    return co2_pressure, co2_loc


def _read_ch4_clim(fname_ch4_clim, date, extent):
    """Return (pressure_hPa, ch4_kg_per_kg) from ERA5/CAMS CH4 HDF5 climatology."""
    sat_mm = date.strftime('%m')
    with h5py.File(fname_ch4_clim, 'r') as f:
        ch4_lon      = f['longitude'][:]
        ch4_lat      = f['latitude'][:]
        ch4_pressure = f['pressure_level'][:]
        ch4_time     = f['valid_time'][:] + 6 * 3600
        ch4_mm       = np.array([int(datetime.datetime.fromtimestamp(t).strftime('%m'))
                                  for t in ch4_time])
        ch4_ind      = np.where(ch4_mm == int(sat_mm))[0]
        if len(ch4_ind) == 0:
            raise ValueError(f"CH4 climatology has no data for month {sat_mm}")

        ch4_clim = f['ch4'][ch4_ind, :, :, :]  # (t, lev, lat, lon) kg/kg

        lat_sort = np.argsort(ch4_lat)
        ch4_lat  = ch4_lat[lat_sort]
        ch4_clim = ch4_clim[:, :, lat_sort, :]

        num_levels = len(ch4_pressure)
        ch4_loc    = np.zeros(num_levels)
        lon_mid    = np.mean(extent[:2])
        lat_mid    = np.mean(extent[2:])

        i0 = max(bisect.bisect_left(ch4_lon, extent[0]) - 2, 0)
        i1 = min(bisect.bisect_left(ch4_lon, extent[1]) + 3, len(ch4_lon))
        j0 = max(bisect.bisect_left(ch4_lat, extent[2]) - 2, 0)
        j1 = min(bisect.bisect_left(ch4_lat, extent[3]) + 3, len(ch4_lat))

        lon_mesh, lat_mesh = np.meshgrid(ch4_lon[i0:i1], ch4_lat[j0:j1])
        clim_mesh = ch4_clim[0, :, j0:j1, i0:i1]

        for i in range(num_levels):
            f_ch4      = interpolate.LinearNDInterpolator(
                             list(zip(lon_mesh.flatten(), lat_mesh.flatten())),
                             clim_mesh[i].flatten())
            ch4_loc[i] = f_ch4(lon_mid, lat_mid)

        mask = ch4_loc > 1
        if not mask.all():
            ch4_loc[mask] = ch4_loc[~mask][0]
        else:
            raise ValueError("CH4 climatology returned all NaN values for this location.")

    return ch4_pressure, ch4_loc


def _read_o3_clim(fname_o3_clim, date, extent):
    """Return (pressure_hPa, o3_kg_per_kg) from MERRA-2 O3 HDF5 climatology."""
    sat_yyyymm = date.strftime('%Y%m')
    with h5py.File(fname_o3_clim, 'r') as f:
        o3_lon      = f['lon'][:]
        o3_lat      = f['lat'][:]
        o3_pressure = f['lev'][:]
        o3_yyyymm   = f['yyyymm'][:]
        o3_ind      = np.where(o3_yyyymm == int(sat_yyyymm))[0]
        if len(o3_ind) == 0:
            raise ValueError(f"O3 climatology has no data for {sat_yyyymm}")

        o3_clim = f['O3'][o3_ind, :, :, :]  # (t, lev, lat, lon) kg/kg
        o3_clim[o3_clim > 0.1] = np.nan

        num_levels = len(o3_pressure)
        o3_loc     = np.zeros(num_levels)
        lon_mid    = np.mean(extent[:2])
        lat_mid    = np.mean(extent[2:])

        i0 = max(bisect.bisect_left(o3_lon, extent[0]) - 2, 0)
        i1 = min(bisect.bisect_left(o3_lon, extent[1]) + 3, len(o3_lon))
        j0 = max(bisect.bisect_left(o3_lat, extent[2]) - 2, 0)
        j1 = min(bisect.bisect_left(o3_lat, extent[3]) + 3, len(o3_lat))

        lon_mesh, lat_mesh = np.meshgrid(o3_lon[i0:i1], o3_lat[j0:j1])
        clim_mesh = o3_clim[0, :, j0:j1, i0:i1]

        for i in range(num_levels):
            f_o3      = interpolate.LinearNDInterpolator(
                            list(zip(lon_mesh.flatten(), lat_mesh.flatten())),
                            clim_mesh[i].flatten())
            o3_loc[i] = f_o3(lon_mid, lat_mid)

        mask = np.logical_or(o3_loc > 1, np.isnan(o3_loc))
        if not mask.all():
            o3_loc[mask] = o3_loc[~mask][0]
        else:
            raise ValueError("O3 climatology returned all NaN values for this location.")

    return o3_pressure, o3_loc


# ─── Atmospheric model builder ─────────────────────────────────────────────────

def _build_atm_profile(zpt_file, fname_std_atm, date, extent,
                        fname_co2_clim=None, fname_ch4_clim=None, fname_o3_clim=None):
    """
    Read a ZPT HDF5 file and build full atmospheric profiles (lev and lay dicts)
    using MODIS data and optional climatology files.  No dropsonde / in-situ /
    MARLi inputs are used.

    Returns
    -------
    lev : dict   – profiles at simulation levels
    lay : dict   – profiles at mid-layer altitudes
    """
    gases = ['o3', 'o2', 'h2o', 'co2', 'no2', 'ch4']

    # ── Read ZPT file ──────────────────────────────────────────────────────────
    with h5py.File(zpt_file, 'r') as hf:
        levels       = hf['level_sim'][...]
        o2mix        = float(hf['o2_mix'][...])
        lev_h2o_vmr  = hf['h2o_vmr'][...]
        lev_h        = hf['h_lev'][...]
        lev_t        = hf['t_lev'][...]
        lev_p        = hf['p_lev'][...]

    layers = 0.5 * (levels[1:] + levels[:-1])

    # ── Read AFGL US standard atmosphere ──────────────────────────────────────
    vnames = ['altitude', 'pressure', 'temperature', 'air',
               'o3', 'o2', 'h2o', 'co2', 'no2']
    units  = ['km', 'mb', 'K', 'cm-3',
               'cm-3', 'cm-3', 'cm-3', 'cm-3', 'cm-3']
    raw = np.genfromtxt(fname_std_atm)

    atm0 = {}
    for i, vname in enumerate(vnames):
        atm0[vname] = {'data': raw[:, i], 'name': vname, 'units': units[i]}

    # Sort ascending by altitude
    idx = np.argsort(atm0['altitude']['data'])
    for key in atm0:
        atm0[key]['data'] = atm0[key]['data'][idx]

    # Convert gas columns from number density to volume mixing ratio
    for key in gases:
        if key in atm0:
            atm0[key]['data']  = atm0[key]['data'] / atm0['air']['data']
            atm0[key]['units'] = 'N/A'

    # ── Check altitude bounds ──────────────────────────────────────────────────
    if levels.min() < atm0['altitude']['data'].min():
        sys.exit('Error [_build_atm_profile]: Input levels too low.')
    if levels.max() > atm0['altitude']['data'].max():
        sys.exit('Error [_build_atm_profile]: Input levels too high.')

    # ── Interpolate to simulation levels/layers ────────────────────────────────
    lev = copy.deepcopy(atm0)
    lev['altitude']['data'] = levels

    lay = copy.deepcopy(atm0)
    lay['altitude']['data'] = layers
    lay['thickness'] = {'name': 'Thickness', 'units': 'km',
                        'data': levels[1:] - levels[:-1]}

    # Standard atmosphere gases (co2, o3, no2 from afglus; overwritten below if clim available)
    for key in ['co2', 'o3', 'no2']:
        f_key = interp1d(atm0['altitude']['data'], atm0[key]['data'],
                         fill_value='extrapolate', kind='linear')
        lev[key]['data'] = f_key(levels)
        lay[key]['data'] = f_key(layers)

    # Temperature and H2O VMR from MODIS
    for key, key_value in zip(['temperature', 'h2o'], [lev_t, lev_h2o_vmr]):
        f_key = interp1d(lev_h, key_value, fill_value='extrapolate', kind='linear')
        lev[key]['data'] = f_key(levels)
        lay[key]['data'] = f_key(layers)

    # Fill any NaN H2O with standard atmosphere
    f_h2o_atm0 = interp1d(atm0['altitude']['data'], atm0['h2o']['data'],
                           fill_value='extrapolate', kind='linear')
    nan_lev = np.isnan(lev['h2o']['data'])
    lev['h2o']['data'][nan_lev] = f_h2o_atm0(levels[nan_lev])
    nan_lay = np.isnan(lay['h2o']['data'])
    lay['h2o']['data'][nan_lay] = f_h2o_atm0(layers[nan_lay])

    # Pressure via barometric formula from MODIS ZPT data
    lev['pressure']['data'] = _atm_interp_pressure(lev_p, lev_h, lev_t, levels, lev['temperature']['data'])
    lay['pressure']['data'] = _atm_interp_pressure(lev_p, lev_h, lev_t, layers, lay['temperature']['data'])

    # Air number density
    kB = 1.380649e-23
    lev['air']['data'] = lev['pressure']['data'] * 100 / (kB * lev['temperature']['data']) * 1e-6
    lay['air']['data'] = lay['pressure']['data'] * 100 / (kB * lay['temperature']['data']) * 1e-6

    # O2 (constant mixing ratio)
    lev['o2']['data'] = o2mix
    lay['o2']['data'] = o2mix

    # ── CO2 climatology  ────────────────────────────────────────────────────────
    # (need to be scaled if want a higher mixing ratio to date)
    if fname_co2_clim is not None:
        co2_pres, co2_vmr = _read_co2_clim(fname_co2_clim, date, extent)
        f_co2 = interp1d(co2_pres, co2_vmr, fill_value='extrapolate', kind='linear')
        lev['co2']['data'] = f_co2(lev['pressure']['data'])
        lay['co2']['data'] = f_co2(lay['pressure']['data'])

    # ── CH4 climatology (or default altitude profile) ──────────────────────────
    if fname_ch4_clim is not None:
        ch4_pres, ch4_kgkg = _read_ch4_clim(fname_ch4_clim, date, extent)
        o2_mw, n2_mw, ch4_mw = 31.998, 28.0134, 16.04
        dry_air_mw = o2_mw * o2mix + n2_mw * (1 - o2mix)
        ch4_vmr = ch4_kgkg / ch4_mw * dry_air_mw
        f_ch4 = interp1d(ch4_pres, ch4_vmr, fill_value='extrapolate', kind='linear')
        lev['ch4'] = {'name': 'ch4', 'units': 'N/A', 'data': f_ch4(lev['pressure']['data'])}
        lay['ch4'] = {'name': 'ch4', 'units': 'N/A', 'data': f_ch4(lay['pressure']['data'])}
    else:
        lev['ch4'] = {'name': 'ch4', 'units': 'N/A', 'data': _atm_interp_ch4(levels)}
        lay['ch4'] = {'name': 'ch4', 'units': 'N/A', 'data': _atm_interp_ch4(layers)}

    # ── O3 climatology ─────────────────────────────────────────────────────────
    if fname_o3_clim is not None:
        o3_pres, o3_kgkg = _read_o3_clim(fname_o3_clim, date, extent)
        o2_mw, n2_mw, o3_mw = 31.998, 28.0134, 47.998
        dry_air_mw = o2_mw * o2mix + n2_mw * (1 - o2mix)
        o3_vmr = o3_kgkg / o3_mw * dry_air_mw
        f_o3 = interp1d(o3_pres, o3_vmr, fill_value='extrapolate', kind='linear')
        lev['o3']['data'] = f_o3(lev['pressure']['data'])
        lay['o3']['data'] = f_o3(lay['pressure']['data'])

    # ── N2O (constant for now) ─────────────────────────────────────────────────────────
    n2o_mix = 0.28e-6
    lev['n2o'] = {'name': 'n2o', 'units': 'N/A', 'data': n2o_mix}
    lay['n2o'] = {'name': 'n2o', 'units': 'N/A', 'data': n2o_mix}

    # ── Convert VMR -> number density [cm-3] ───────────────────────────────────
    lev['factor'] = {'name': 'number density factor', 'units': 'cm-3',
                     'data': 6.02214179e23 / 8.314472 * lev['pressure']['data']
                             / lev['temperature']['data'] * 1e-4}
    lay['factor'] = {'name': 'number density factor', 'units': 'cm-3',
                     'data': 6.02214179e23 / 8.314472 * lay['pressure']['data']
                             / lay['temperature']['data'] * 1e-4}

    for key in gases:
        if key in lev:
            lev[key]['data']  = lev[key]['data'] * lev['factor']['data']
            lev[key]['units'] = 'cm-3'
            lay[key]['data']  = lay[key]['data'] * lay['factor']['data']
            lay[key]['units'] = 'cm-3'

    return lev, lay


# ─── Public function ───────────────────────────────────────────────────────────

def prepare_atmospheric_profile(fdir_data, case_tag, date,
                                 sfc_alt_avg,
                                 fname_mod07,
                                 levels=None,
                                 mod_extent=None,
                                 zpt_filedir='./data/atmospheric_profiles',
                                 fname_std_atm=None,
                                 sfc_T=None,
                                 sfc_h_to_zero=True,
                                 plot=False):
    """
    Prepare an atmospheric profile for LW radiative transfer simulations.

    Unlike the LRT version this function:
      - Accepts the MODIS MOD-07 file(s) directly (no download).
      - Does not use dropsonde, in-situ, or MARLi water-vapour data.

    Parameters
    ----------
    fdir_data : str
        Root data directory (must contain ``climatology/`` sub-folder).
    case_tag : str
        Case identifier used in output filenames.
    date : datetime.date or datetime.datetime
        Observation date (used for climatology month look-up).
    sfc_alt_avg : float
        Average aircraft altitude in km (used only for output filenames).
    fname_mod07 : str
        Path to the MOD07/MYD07 HDF4 file.
    levels : array-like or None
        Simulation altitude levels in km.  None uses a default grid.
    mod_extent : list or None
        [lon_west, lon_east, lat_south, lat_north].
        Defaults to [-60.0, -80.0, 82.4, 84.6].
    zpt_filedir : str
        Directory for intermediate ZPT files and output ASCII profiles.
    fname_std_atm : str or None
        Path to the AFGL US standard atmosphere file (``afglus.dat``).
        Defaults to the copy shipped with er3t.
    sfc_T : float or None
        Override surface temperature (K).
    sfc_h_to_zero : bool
        Force surface height to 0 km.

    Output files written to ``zpt_filedir``
    ----------------------------------------
    ``atm_profiles_{date_s}_{case_tag}_{sfc_alt_avg:.2f}km.dat``
        Full profile: z, p, T, air, O3, O2, H2O, CO2, NO2.
    ``ch4_profiles_{date_s}_{case_tag}_{sfc_alt_avg:.2f}km.dat``
        CH4-only profile: z, CH4.
    """
    if mod_extent is None:
        mod_extent = [-80.0, -60.0, 82.4, 84.6]
    # Normalise to [lon_min, lon_max, lat_min, lat_max]
    mod_extent = [np.float64(min(mod_extent[0], mod_extent[1])),
                  np.float64(max(mod_extent[0], mod_extent[1])),
                  np.float64(min(mod_extent[2], mod_extent[3])),
                  np.float64(max(mod_extent[2], mod_extent[3]))]

    if fname_std_atm is None:
        # Fall back to the er3t data directory (no er3t import needed at runtime)
        _this_dir   = os.path.dirname(os.path.abspath(__file__))
        fname_std_atm = os.path.join(_this_dir,
                                    '../../../../er3t/er3t/er3t/data/atmmod/afglus.dat')
        fname_std_atm = os.path.normpath(fname_std_atm)
        if not os.path.exists(fname_std_atm):
            raise FileNotFoundError(
                "afglus.dat not found at the default location. "
                "Please supply fname_std_atm explicitly.")

    date_s = date.strftime('%Y%m%d')

    os.makedirs(zpt_filedir, exist_ok=True)

    zpt_filename = f'zpt_{date_s}_{case_tag}.h5'
    zpt_path     = os.path.join(zpt_filedir, zpt_filename)

    # Build ZPT file from MODIS
    status = _create_modis_atm_zpt(
        fname_mod07   = fname_mod07,
        extent        = mod_extent,
        output_dir    = zpt_filedir,
        output        = zpt_filename,
        levels        = levels,
        sfc_T_set     = sfc_T,
        sfc_h_to_zero = sfc_h_to_zero,
    )
    if status != 'success':
        raise RuntimeError("Failed to create ZPT file from MODIS data.")

    # Build atmospheric model
    lev, lay = _build_atm_profile(
        zpt_file       = zpt_path,
        fname_std_atm   = fname_std_atm,
        date           = date,
        extent         = mod_extent,
        fname_co2_clim = os.path.join(fdir_data, 'climatology',
                                      'cams73_latest_co2_conc_surface_inst_2020.nc'),
        fname_ch4_clim = os.path.join(fdir_data, 'climatology',
                                      'cams_ch4_202005-202008.nc'),
        fname_o3_clim  = os.path.join(fdir_data, 'climatology',
                                      'ozone_merra2_202405_202408.h5'),
    )

    if np.any(np.isnan(lev['pressure']['data'])):
        raise ValueError("NaN values found in pressure profile. "
                         "Check MODIS 07 data coverage for the given extent.")

    nz = len(lev['altitude']['data'])

    # Write full atmospheric profile
    atm_fname = os.path.join(
        zpt_filedir,
        f'atm_profiles_{date_s}_{case_tag}_{sfc_alt_avg:.2f}km.dat'
    )
    with open(atm_fname, 'w') as f:
        header = ('# Combined atmospheric profile\n'
                  '#      z(km)      p(mb)        T(K)    air(cm-3)    o3(cm-3)'
                  '     o2(cm-3)    h2o(cm-3)    co2(cm-3)     no2(cm-3)\n')
        lines = [
            f'{lev["altitude"]["data"][i]:11.3f} {lev["pressure"]["data"][i]:11.5f} '
            f'{lev["temperature"]["data"][i]:11.3f} '
            f'{lev["air"]["data"][i]:12.6e} {lev["o3"]["data"][i]:12.6e} '
            f'{lev["o2"]["data"][i]:12.6e} {lev["h2o"]["data"][i]:12.6e} '
            f'{lev["co2"]["data"][i]:12.6e} {lev["no2"]["data"][i]:12.6e}'
            for i in range(nz)[::-1]
        ]
        f.write(header + '\n'.join(lines))

    # Write CH4-only profile
    ch4_fname = os.path.join(
        zpt_filedir,
        f'ch4_profiles_{date_s}_{case_tag}_{sfc_alt_avg:.2f}km.dat'
    )
    with open(ch4_fname, 'w') as f:
        header = ('# Combined atmospheric profile for ch4 only\n'
                  '#      z(km)      ch4(cm-3)\n')
        lines = [
            f'{lev["altitude"]["data"][i]:11.3f} {lev["ch4"]["data"][i]:12.6e}'
            for i in range(nz)[::-1]
        ]
        f.write(header + '\n'.join(lines))

    if plot:
        from metpy.calc import saturation_vapor_pressure, dewpoint_from_relative_humidity
        from metpy.plots import SkewT
        from metpy.units import units
        import matplotlib.pyplot as plt

        p      = lev['pressure']['data']
        T      = lev['temperature']['data']
        h2o    = lev['h2o']['data'] / lev['air']['data']  # cm-3 -> VMR

        es      = saturation_vapor_pressure(T * units.kelvin).to('hPa')
        p_water = p * h2o * units.hPa
        rh     = p_water / es
        rh[rh > 1] = 1
        dewT = dewpoint_from_relative_humidity(T * units.kelvin, rh).to('kelvin')

        p_prf  = p * units.hPa
        T_prf  = (T  * units.kelvin).to(units.degC)
        Td_prf = (dewT.magnitude * units.kelvin).to(units.degC)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6.75))
        ax1.set_visible(False)
        skew = SkewT(fig=fig, subplot=(1, 2, 1), aspect=120.5)
        skew.plot(p_prf, T_prf,  'r', label='Temperature', linewidth=3)
        skew.plot(p_prf, Td_prf, 'g', label='Dew Point',   linewidth=3)
        skew.ax.set_xlabel('Temperature (°C)', fontsize=14)
        skew.ax.set_ylabel('Pressure (hPa)',   fontsize=14)
        skew.plot_dry_adiabats()
        skew.plot_moist_adiabats()
        skew.plot_mixing_lines()
        skew.ax.set_ylim(1000, 100)
        skew.ax.legend()

        ax2.plot(h2o, p_prf, 'b:', linewidth=3, label='H$_2$O')
        ax2.set_yscale('log')
        ax2.set_ylim(1000, 100)
        ax2.set_xlabel('H$_2$O VMR', fontsize=14)
        ax2.set_ylabel('Pressure (hPa)', fontsize=14)
        ax2.legend()
        ax2.grid(True)

        fig.tight_layout()
        plot_fname = os.path.join(zpt_filedir, f'skewt_{date_s}_{case_tag}.png')
        fig.savefig(plot_fname, dpi=300)
        plt.close(fig)

    return None


if __name__ == '__main__':
    pass
