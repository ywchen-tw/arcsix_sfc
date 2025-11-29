import os
import sys
import glob
import copy
import time
from collections import OrderedDict
import datetime
import multiprocessing as mp
import pickle
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
import h5py
from netCDF4 import Dataset
import numpy as np
from scipy.interpolate import RegularGridInterpolator, griddata, NearestNDInterpolator
from scipy import ndimage
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path as mpl_path
import matplotlib.image as mpl_img
import matplotlib.axes as maxes
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib import rcParams, ticker
from matplotlib.ticker import FixedLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy
import cartopy.crs as ccrs
import bisect
import pandas as pd
from scipy.signal import convolve

import xarray as xr
from collections import defaultdict
# mpl.use('Agg')


import er3t

def gaussian(x, mu, sig):
    y = (
        1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
    )
    
    return y/np.max(y)


def read_ict_radiosonde(filepath, encoding='utf-8', na_values=None):
    """
    Reads an ICT file but only loads the columns:
      Time_Start, Time_Stop, WD, H, P, RH, WS, T
    
    Returns:
      header_lines: list of strings (the metadata/header)
      df: pandas DataFrame of just those eight fields
    """
    if na_values is None:
        na_values = [-9999999, -888, -777]

    # the only columns we care about
    target = ["Time_Start", "Time_Stop", "WD", "H", "P", "RH", "WS", "T"]

    with open(filepath, 'r', encoding=encoding, errors='replace') as f:
        header_lines = []
        for raw in f:
            line = raw.strip()
            if line.startswith('Time_Start,Time_Stop,WD,H,P,RH,WS,T'):
                raw_cols = [c.strip() for c in line.split(',')]
                break
            header_lines.append(raw)

        # mangle duplicates exactly as before
        from collections import defaultdict
        counter = defaultdict(int)
        cols = []
        for name in raw_cols:
            if counter[name]:
                name = f"{name}_{counter[name]}"
            cols.append(name)
            counter[name] += 1

        # figure out which of our target cols actually ended up in `cols`
        usecols = [c for c in cols if c in target]

        # now read only those columns
        df = pd.read_csv(
            f,
            header=None,
            names=cols,
            usecols=usecols,
            na_values=na_values,
            comment='#'
        )
        
        # calculate Dew Point and Relative Humidity
        import metpy.calc as mpcalc
        from metpy.units import units
        from metpy.calc import dewpoint_from_relative_humidity
        # convert columns to MetPy units
        p = df['P'].values * units.hPa
        t = df['T'].values * units.degC
        rh = df['RH'].values * units.percent
        h = df['H'].values * units.meter
        ws = df['WS'].values * units.meter / units.second
        # calculate dew point temperature
        t_dew = dewpoint_from_relative_humidity(t, rh) # in degC
        # print("t_dew:", t_dew)
        # sys.exit()
        
        # calculate water vapor mixing ratio
        # using the formula: w = 0.622 * (e / (p - e))
        # where e is the vapor pressure, which can be calculated from RH and P
        h2o_mr = mpcalc.mixing_ratio_from_relative_humidity(p, t, rh).to('g/kg')
        
        # convert to DataFrame
        radiosonde_df = pd.DataFrame({'p': df['P'], 
                                        't_dry': df['T']+273.15, 
                                        'rh': df['RH'], 
                                        'alt': df['H'],
                                        't_dew': np.array(t_dew)+273.15, 
                                        'h2o_mr': np.array(h2o_mr), 
                                        'ws': ws})

    radiosonde_df = radiosonde_df[~radiosonde_df['p'].isna()]

    return header_lines, radiosonde_df

def read_ict_dropsonde(filepath,
                          encoding='utf-8',
                          na_values=None,
                          **read_csv_kwargs):
    """
    Reads an ICT file but only returns the data lines *after* the given marker line.
    
    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to your .ict file.
    marker : str
        The exact line at which to stop skipping. Everything after this line is read in.
    encoding : str
        File encoding (passed to open).
    **read_csv_kwargs : dict
        Additional keyword args passed to pd.read_csv (e.g. na_values, comment).
    
    Returns
    -------
    header_lines : list of str
        All lines up through (but not including) the marker.
    df : pandas.DataFrame
        Data read from all lines after the marker.
    """
    if na_values is None:
        na_values = [-9999999, -888, -777]
    
    # the only columns we care about
    target = ["Time_Start", "ElaspedTime", "Pressure", "Temperature", "RH", 
              "WindSpeed", "WindDir", "Latitude", "Longitude",
              "Pressure_Altitude", "GPS_Altitude", "Dewpoint",
              "U_wind", "V_wind", "W_wind", "DescentRate", "MixingRatio",
              "Theta", "Theta_e", "Theta_v", "T_virtual"]\
    
    with open(filepath, 'r', encoding=encoding, errors='replace') as f:
        header_lines = []
        for raw in f:
            line = raw.strip()
            if line.startswith( "Time_Start,ElaspedTime,Pressure,Temperature,RH,WindSpeed,WindDir,Latitude,Longitude,Pressure_Altitude,GPS_Altitude,Dewpoint,U_wind,V_wind,W_wind,DescentRate,MixingRatio,Theta,Theta_e,Theta_v,T_virtual"):
                raw_cols = [c.strip() for c in line.split(',')]
                break
            header_lines.append(raw)

        # mangle duplicates exactly as before
        from collections import defaultdict
        counter = defaultdict(int)
        cols = []
        for name in raw_cols:
            if counter[name]:
                name = f"{name}_{counter[name]}"
            cols.append(name)
            counter[name] += 1

        # figure out which of our target cols actually ended up in `cols`
        usecols = [c for c in cols if c in target]

        # now read only those columns
        df = pd.read_csv(
            f,
            header=None,
            names=cols,
            usecols=usecols,
            na_values=na_values,
            comment='#'
        )
        
        df.loc[df['Pressure'] < 0, 'Pressure'] = np.nan
        df.loc[df['Temperature'] < -273, 'Temperature'] = np.nan
        df.loc[df['RH'] < 0, 'RH'] = np.nan
        df.loc[df['WindSpeed'] < 0, 'WindSpeed'] = np.nan
        df.loc[df['Dewpoint'] < -273, 'Dewpoint'] = np.nan
        df.loc[df['MixingRatio'] < 0, 'MixingRatio'] = np.nan
        
        df = df.dropna(subset=['Pressure', 'Temperature', 'RH', 'WindSpeed', 'Dewpoint', 'MixingRatio'])
        
        # convert to DataFrame
        dropsonde_df = pd.DataFrame({'time': df['Time_Start']/3600.0,
                                        'p': df['Pressure'], 
                                        't_dry': df['Temperature']+273.15, 
                                        'rh': df['RH'], 
                                        'alt': df['GPS_Altitude'],
                                        't_dew': df['Dewpoint']+273.15, 
                                        'h2o_mr': df['MixingRatio'], # in g/kg
                                        'ws': df['WindSpeed'],
                                        'lon_all': df['Longitude'],
                                        'lat_all': df['Latitude'],})

    dropsonde_df = dropsonde_df[~dropsonde_df['p'].isna()]

    return header_lines, dropsonde_df

def read_ict_lwc(filepath, encoding='utf-8', na_values=None):
    """
    Reads an ICT file but only loads the columns:
      Time_Start, TWC_wcm42, Lwc1_wcm42, Lwc2_wcm42
    
    Returns:
      header_lines: list of strings (the metadata/header)
      df: pandas DataFrame of just those four fields
    """
    if na_values is None:
        na_values = [-9999999, -888, -777]

    # the only columns we care about
    target = ["Time_Start", "TWC_wcm42", "Lwc1_wcm42", "Lwc2_wcm42"]

    with open(filepath, 'r', encoding=encoding, errors='replace') as f:
        header_lines = []
        for raw in f:
            line = raw.strip()
            if line.startswith('Time_Start,TWC_wcm42,Lwc1_wcm42,Lwc2_wcm42'):
                raw_cols = [c.strip() for c in line.split(',')]
                break
            header_lines.append(raw)

        # mangle duplicates exactly as before
        from collections import defaultdict
        counter = defaultdict(int)
        cols = []
        for name in raw_cols:
            if counter[name]:
                name = f"{name}_{counter[name]}"
            cols.append(name)
            counter[name] += 1

        # figure out which of our target cols actually ended up in `cols`
        usecols = [c for c in cols if c in target]

        # now read only those columns
        df = pd.read_csv(
            f,
            header=None,
            names=cols,
            usecols=usecols,
            na_values=na_values,
            comment='#'
        )
        
        df.loc[df['TWC_wcm42'] < 0, 'TWC_wcm42'] = np.nan
        df.loc[df['Lwc1_wcm42'] < 0, 'Lwc1_wcm42'] = np.nan
        df.loc[df['Lwc2_wcm42'] < 0, 'Lwc2_wcm42'] = np.nan
        df = df.dropna(subset=['TWC_wcm42', 'Lwc1_wcm42', 'Lwc2_wcm42'])
        
        # convert to DataFrame
        LWC_df = pd.DataFrame({'tmhr': df['Time_Start']/86400*24.0, 
                                'TWC': df['TWC_wcm42'], 
                                'LWC_1': df['Lwc1_wcm42'], 
                                'LWC_2': df['Lwc2_wcm42'],})

    LWC_df = LWC_df[~LWC_df['LWC_1'].isna()]

    return header_lines, LWC_df


def read_ict_cloud_micro_2DGRAY50(filepath, encoding='utf-8', na_values=None):
    """
    Reads an ICT file but only loads the columns:
      Time_Start,sv,conc,ext,iwc,acceptPCNT,meanDiameter,meanVolDiam,effectiveDiam....
      
    Returns:
      header_lines: list of strings (the metadata/header)
      df: pandas DataFrame of just those four fields
    """
    if na_values is None:
        na_values = [-9999999, -888, -777]

    # the only columns we care about
    target = ["Time_Start", "sv", "conc", "ext", "iwc", "acceptPCNT", "meanDiameter", "meanVolDiam", "effectiveDiam", 
              "cbin01", "cbin02", "cbin03", "cbin04", "cbin05", "cbin06", "cbin07", "cbin08", "cbin09", "cbin10", 
              "cbin11", "cbin12", "cbin13", "cbin14", "cbin15", "cbin16", "cbin17", "cbin18", "cbin19", "cbin20", 
              "cbin21", "cbin22", "cbin23", "cbin24", "cbin25", "cbin26", "cbin27", "cbin28", "cbin29", "cbin30", 
              "cbin31", "cbin32", "cbin33", "cbin34", "cbin35", "cbin36", "cbin37", "cbin38", "cbin39", "cbin40", 
              "cbin41", "cbin42", "cbin43", "cbin44", "cbin45", "cbin46", "cbin47", "cbin48", "cbin49", "cbin50", 
              "cbin51", "cbin52", "cbin53", "cbin54", "cbin55", "cbin56", "cbin57", "cbin58", "cbin59", "cbin60", 
              "cbin61", 
              "abin01", "abin02", "abin03", "abin04", "abin05", "abin06", "abin07", "abin08", "abin09", "abin10", 
              "abin11", "abin12", "abin13", "abin14", "abin15", "abin16", "abin17", "abin18", "abin19", "abin20",
              "abin21", "abin22", "abin23", "abin24", "abin25", "abin26", "abin27", "abin28", "abin29", "abin30",
              "abin31", "abin32", "abin33", "abin34", "abin35", "abin36", "abin37", "abin38", "abin39", "abin40", 
              "abin41", "abin42", "abin43", "abin44", "abin45", "abin46", "abin47", "abin48", "abin49", "abin50", 
              "abin51", "abin52", "abin53", "abin54", "abin55", "abin56", "abin57", "abin58", "abin59", "abin60", 
              "abin61", 
              "mbin01", "mbin02", "mbin03", "mbin04", "mbin05", "mbin06", "mbin07", "mbin08", "mbin09", "mbin10", 
              "mbin11", "mbin12", "mbin13", "mbin14", "mbin15", "mbin16", "mbin17", "mbin18", "mbin19", "mbin20", 
              "mbin21", "mbin22", "mbin23", "mbin24", "mbin25", "mbin26", "mbin27", "mbin28", "mbin29", "mbin30", 
              "mbin31", "mbin32", "mbin33", "mbin34", "mbin35", "mbin36", "mbin37", "mbin38", "mbin39", "mbin40", 
              "mbin41", "mbin42", "mbin43", "mbin44", "mbin45", "mbin46", "mbin47", "mbin48", "mbin49", "mbin50", 
              "mbin51", "mbin52", "mbin53", "mbin54", "mbin55", "mbin56", "mbin57", "mbin58", "mbin59", "mbin60", 
              "mbin61", 
              "nbin01", "nbin02", "nbin03", "nbin04", "nbin05", "nbin06", "nbin07", "nbin08", "nbin09", "nbin10", 
              "nbin11", "nbin12", "nbin13", "nbin14", "nbin15", "nbin16", "nbin17", "nbin18", "nbin19", "nbin20", 
              "nbin21", "nbin22", "nbin23", "nbin24", "nbin25", "nbin26", "nbin27", "nbin28", "nbin29", "nbin30", 
              "nbin31", "nbin32", "nbin33", "nbin34", "nbin35", "nbin36", "nbin37", "nbin38", "nbin39", "nbin40", 
              "nbin41", "nbin42", "nbin43", "nbin44", "nbin45", "nbin46", "nbin47", "nbin48", "nbin49", "nbin50", 
              "nbin51", "nbin52", "nbin53", "nbin54", "nbin55", "nbin56", "nbin57", "nbin58", "nbin59", "nbin60", 
              "nbin61", "totaln"]

    with open(filepath, 'r', encoding=encoding, errors='replace') as f:
        header_lines = []
        for raw in f:
            line = raw.strip()
            if line.startswith('Time_Start,sv,conc,ext,iwc,acceptPCNT,meanDiameter,meanVolDiam,effectiveDiam'):
                raw_cols = [c.strip() for c in line.split(',')]
                break
            header_lines.append(raw)

        # mangle duplicates exactly as before
        from collections import defaultdict
        counter = defaultdict(int)
        cols = []
        for name in raw_cols:
            if counter[name]:
                name = f"{name}_{counter[name]}"
            cols.append(name)
            counter[name] += 1

        # figure out which of our target cols actually ended up in `cols`
        usecols = [c for c in cols if c in target]

        # now read only those columns
        df = pd.read_csv(
            f,
            header=None,
            names=cols,
            usecols=usecols,
            na_values=na_values,
            comment='#'
        )
        
        df.loc[df['conc'] < 0, 'conc'] = np.nan
        df.loc[df['ext'] < 0, 'ext'] = np.nan
        df.loc[df['iwc'] < 0, 'iwc'] = np.nan
        df.loc[df['effectiveDiam'] < 0, 'effectiveDiam'] = np.nan
        # df = df.dropna(subset=['conc', 'ext', 'iwc', 'effectiveDiam'])
        
        # convert to DataFrame
        cloud_df = pd.DataFrame({'tmhr': df['Time_Start']/86400*24.0, 
                               'conc': df['conc'],
                               'ext': df['ext'], 
                               'iwc': df['iwc'], 
                               'effectiveDiam': df['effectiveDiam'],})

    cloud_df = cloud_df[~cloud_df['ext'].isna()]

    return header_lines, cloud_df


def read_ict_cloud_micro_FCDP(filepath, encoding='utf-8', na_values=None):
    """
    Reads an ICT file but only loads the columns:
      Time_Start,sv,conc,ext,iwc,acceptPCNT,meanDiameter,meanVolDiam,effectiveDiam....
      
    Returns:
      header_lines: list of strings (the metadata/header)
      df: pandas DataFrame of just those four fields
    """
    if na_values is None:
        na_values = [-9999999, -888, -777]

    # the only columns we care about
    target = ["Time_Start", "conc", "ext", "lwc", "sv", "CNT", 
              "cbin01", "cbin02", "cbin03", "cbin04", "cbin05", "cbin06", "cbin07", "cbin08", "cbin09", "cbin10", 
              "cbin11", "cbin12", "cbin13", "cbin14", "cbin15", "cbin16", "cbin17", "cbin18", "cbin19", "cbin20", "cbin21", 
              "nbin01", "nbin02", "nbin03", "nbin04", "nbin05", "nbin06", "nbin07", "nbin08", "nbin09", "nbin10", 
              "nbin11", "nbin12", "nbin13", "nbin14", "nbin15", "nbin16", "nbin17", "nbin18", "nbin19", "nbin20", "nbin21"]

    with open(filepath, 'r', encoding=encoding, errors='replace') as f:
        header_lines = []
        for raw in f:
            line = raw.strip()
            if line.startswith('Time_Start,conc,ext,lwc,sv,CNT,cbin01,cbin02,cbin03,cbin04,cbin05,cbin06,cbin07,cbin08,cbin09,cbin10,cbin11,cbin12,cbin13,cbin14,cbin15,cbin16,cbin17,cbin18,cbin19,cbin20,cbin21,nbin01,nbin02,nbin03,nbin04,nbin05,nbin06,nbin07,nbin08,nbin09,nbin10,nbin11,nbin12,nbin13,nbin14,nbin15,nbin16,nbin17,nbin18,nbin19,nbin20,nbin21'):
                raw_cols = [c.strip() for c in line.split(',')]
                break
            header_lines.append(raw)

        # mangle duplicates exactly as before
        from collections import defaultdict
        counter = defaultdict(int)
        cols = []
        for name in raw_cols:
            if counter[name]:
                name = f"{name}_{counter[name]}"
            cols.append(name)
            counter[name] += 1

        # figure out which of our target cols actually ended up in `cols`
        usecols = [c for c in cols if c in target]

        # now read only those columns
        df = pd.read_csv(
            f,
            header=None,
            names=cols,
            usecols=usecols,
            na_values=na_values,
            comment='#'
        )
        
        df.loc[df['conc'] < 0, 'conc'] = np.nan
        df.loc[df['ext'] < 0, 'ext'] = np.nan
        df.loc[df['lwc'] < 0, 'lwc'] = np.nan
        # df = df.dropna(subset=['conc', 'ext', 'iwc', 'effectiveDiam'])
        
        # convert to DataFrame
        cloud_df = pd.DataFrame({'tmhr': df['Time_Start']/86400*24.0, 
                               'conc': df['conc'],
                               'ext': df['ext'], 
                               'lwc': df['lwc'],})

    cloud_df = cloud_df[~cloud_df['ext'].isna()]

    return header_lines, cloud_df

def read_ict_bbr(filepath, encoding='utf-8', na_values=None):
    """
    Reads an ICT file but only loads the columns:
      Time_Start,  DN_IR_Irrad,  UP_IR_Irrad,  DN_CM22_SOLAR_Irrad,  UP_CM22_SOLAR_Irrad,  DN_SPN1_Total_SOLAR_Irrad, DN_SPN1_Diffuse_SOLAR_Irrad, IR_Sky_Temp
      
    Returns:
      header_lines: list of strings (the metadata/header)
      df: pandas DataFrame of just those four fields
    """
    if na_values is None:
        na_values = [-9999999, -888, -777]

    # the only columns we care about
    target = ["Time_Start", "DN_IR_Irrad", "UP_IR_Irrad", 
              "DN_CM22_SOLAR_Irrad", "UP_CM22_SOLAR_Irrad", 
              "DN_SPN1_Total_SOLAR_Irrad", "DN_SPN1_Diffuse_SOLAR_Irrad", "IR_Sky_Temp"]

    with open(filepath, 'r', encoding=encoding, errors='replace') as f:
        header_lines = []
        for raw in f:
            line = raw.strip()
            if line.startswith('Time_Start,  DN_IR_Irrad,  UP_IR_Irrad,  DN_CM22_SOLAR_Irrad,  UP_CM22_SOLAR_Irrad,  DN_SPN1_Total_SOLAR_Irrad, DN_SPN1_Diffuse_SOLAR_Irrad, IR_Sky_Temp'):
                raw_cols = [c.strip() for c in line.split(',')]
                break
            header_lines.append(raw)

        # mangle duplicates exactly as before
        from collections import defaultdict
        counter = defaultdict(int)
        cols = []
        for name in raw_cols:
            if counter[name]:
                name = f"{name}_{counter[name]}"
            cols.append(name)
            counter[name] += 1

        # figure out which of our target cols actually ended up in `cols`
        usecols = [c for c in cols if c in target]

        # now read only those columns
        df = pd.read_csv(
            f,
            header=None,
            names=cols,
            usecols=usecols,
            na_values=na_values,
            comment='#'
        )
        
        df.loc[df['DN_IR_Irrad'] < 0, 'DN_IR_Irrad'] = np.nan
        df.loc[df['UP_IR_Irrad'] < 0, 'UP_IR_Irrad'] = np.nan
        df.loc[df['DN_CM22_SOLAR_Irrad'] < 0, 'DN_CM22_SOLAR_Irrad'] = np.nan
        df.loc[df['UP_CM22_SOLAR_Irrad'] < 0, 'UP_CM22_SOLAR_Irrad'] = np.nan
        df.loc[df['DN_SPN1_Total_SOLAR_Irrad'] < 0, 'DN_SPN1_Total_SOLAR_Irrad'] = np.nan
        df.loc[df['DN_SPN1_Diffuse_SOLAR_Irrad'] < 0, 'DN_SPN1_Diffuse_SOLAR_Irrad'] = np.nan
        # df = df.dropna(subset=['conc', 'ext', 'iwc', 'effectiveDiam'])
        
        # convert to DataFrame
        bbr_df = pd.DataFrame({'tmhr': df['Time_Start']/86400*24.0, 
                               'DN_IR_Irrad': df['DN_IR_Irrad'],
                               'UP_IR_Irrad': df['UP_IR_Irrad'],
                               'DN_CM22_SOLAR_Irrad': df['DN_CM22_SOLAR_Irrad'], 
                               'UP_CM22_SOLAR_Irrad': df['UP_CM22_SOLAR_Irrad'],
                               'DN_SPN1_Total_SOLAR_Irrad': df['DN_SPN1_Total_SOLAR_Irrad'],
                               'DN_SPN1_Diffuse_SOLAR_Irrad': df['DN_SPN1_Diffuse_SOLAR_Irrad'],
                               'IR_Sky_Temp': df['IR_Sky_Temp'],  # in degC
                               })

    # bbr_df = bbr_df[~bbr_df['DN_IR_Irrad'].isna()]

    return header_lines, bbr_df

def read_ict_kt19(filepath, encoding='utf-8', na_values=None):
    """
    Reads an ICT file but only loads the columns:
      Time_Start,Latitude,Longitude,GPS_Altitude,IR_Surf_Temp
      
    Returns:
      header_lines: list of strings (the metadata/header)
      df: pandas DataFrame of just those four fields
    """
    if na_values is None:
        na_values = [-9999999, -888, -777]

    # the only columns we care about
    target = ["Time_Start", "Latitude", "Longitude", "GPS_Altitude", "IR_Surf_Temp"]

    with open(filepath, 'r', encoding=encoding, errors='replace') as f:
        header_lines = []
        for raw in f:
            line = raw.strip()
            if line.startswith('Time_Start,Latitude,Longitude,GPS_Altitude,IR_Surf_Temp'):
                raw_cols = [c.strip() for c in line.split(',')]
                break
            header_lines.append(raw)

        # mangle duplicates exactly as before
        from collections import defaultdict
        counter = defaultdict(int)
        cols = []
        for name in raw_cols:
            if counter[name]:
                name = f"{name}_{counter[name]}"
            cols.append(name)
            counter[name] += 1

        # figure out which of our target cols actually ended up in `cols`
        usecols = [c for c in cols if c in target]

        # now read only those columns
        df = pd.read_csv(
            f,
            header=None,
            names=cols,
            usecols=usecols,
            na_values=na_values,
            comment='#'
        )
        
        df.loc[df['GPS_Altitude'] < 0, 'GPS_Altitude'] = np.nan
        df.loc[df['IR_Surf_Temp'] < -273.15, 'IR_Surf_Temp'] = np.nan
        # df = df.dropna(subset=['conc', 'ext', 'iwc', 'effectiveDiam'])
        
        # convert to DataFrame
        kt19_df = pd.DataFrame({'tmhr': df['Time_Start']/86400*24.0, 
                               'lon': df['Longitude'],
                               'lat': df['Latitude'],
                               'alt': df['GPS_Altitude'], 
                               'ir_sfc_T': df['IR_Surf_Temp'], # in degC
                               })

    # kt19_df = kt19_df[~kt19_df['ir_sfc_T'].isna()]

    return header_lines, kt19_df

def read_ict_dlh_h2o(filepath, encoding='utf-8', na_values=None):
    """
    Reads an ICT file but only loads the columns:
      Time_Start,H2O_DLH,RHi_DLH,RHw_DLH
      
    Returns:
      header_lines: list of strings (the metadata/header)
      df: pandas DataFrame of just those four fields
    """
    if na_values is None:
        na_values = [-9999999, -8888, -7777]

    # the only columns we care about
    target = ["Time_Start", "H2O_DLH", "RHi_DLH", "RHw_DLH"]

    with open(filepath, 'r', encoding=encoding, errors='replace') as f:
        header_lines = []
        for raw in f:
            line = raw.strip()
            if line.startswith('Time_Start,H2O_DLH,RHi_DLH,RHw_DLH'):
                raw_cols = [c.strip() for c in line.split(',')]
                break
            header_lines.append(raw)

        # mangle duplicates exactly as before
        from collections import defaultdict
        counter = defaultdict(int)
        cols = []
        for name in raw_cols:
            if counter[name]:
                name = f"{name}_{counter[name]}"
            cols.append(name)
            counter[name] += 1

        # figure out which of our target cols actually ended up in `cols`
        usecols = [c for c in cols if c in target]

        # now read only those columns
        df = pd.read_csv(
            f,
            header=None,
            names=cols,
            usecols=usecols,
            na_values=na_values,
            comment='#'
        )
        
        for var in ['H2O_DLH', 'RHi_DLH', 'RHw_DLH']:
            df.loc[df[var] < 0, var] = np.nan
        
        # convert to DataFrame
        dlh_h2o_df = pd.DataFrame({'tmhr': df['Time_Start']/86400*24.0, 
                               'h2o_vmr': df['H2O_DLH'], # in ppmv
                               'RHi_DLH': df['RHi_DLH'], # RH with respect to ice
                               'RHw_DLH': df['RHw_DLH'], # RH with respect to water
                               })

    # kt19_df = kt19_df[~kt19_df['ir_sfc_T'].isna()]

    return header_lines, dlh_h2o_df


def read_ict_ch4(filepath, encoding='utf-8', na_values=None):
    """
    Reads an ICT file but only loads the columns:
      Time_Start,Time_Stop,Time_Mid,CH4_ppm
      
    Returns:
      header_lines: list of strings (the metadata/header)
      df: pandas DataFrame of just those four fields
    """
    if na_values is None:
        na_values = [-9999, -8888, -7777]

    # the only columns we care about
    target = ["Time_Start", "Time_Stop", "Time_Mid", "CH4_ppm"]

    with open(filepath, 'r', encoding=encoding, errors='replace') as f:
        header_lines = []
        for raw in f:
            line = raw.strip()
            if line.startswith('Time_Start,Time_Stop,Time_Mid,CH4_ppm'):
                raw_cols = [c.strip() for c in line.split(',')]
                break
            header_lines.append(raw)

        # mangle duplicates exactly as before
        from collections import defaultdict
        counter = defaultdict(int)
        cols = []
        for name in raw_cols:
            if counter[name]:
                name = f"{name}_{counter[name]}"
            cols.append(name)
            counter[name] += 1

        # figure out which of our target cols actually ended up in `cols`
        usecols = [c for c in cols if c in target]

        # now read only those columns
        df = pd.read_csv(
            f,
            header=None,
            names=cols,
            usecols=usecols,
            na_values=na_values,
            comment='#'
        )
        
        df.loc[df['CH4_ppm'] < 0, 'CH4_ppm'] = np.nan
        
        # convert to DataFrame
        ch4_df = pd.DataFrame({'tmhr': df['Time_Mid']/86400*24.0, 
                               'ch4': df['CH4_ppm'],
                               })
        
    return header_lines, ch4_df


def read_ict_co(filepath, encoding='utf-8', na_values=None):
    """
    Reads an ICT file but only loads the columns:
      Time_Start,Time_Stop,Time_Mid,CO_ppm
      
    Returns:
      header_lines: list of strings (the metadata/header)
      df: pandas DataFrame of just those four fields
    """
    if na_values is None:
        na_values = [-9999, -8888, -7777]

    # the only columns we care about
    target = ["Time_Start", "Time_Stop", "Time_Mid", "CO_ppm"]

    with open(filepath, 'r', encoding=encoding, errors='replace') as f:
        header_lines = []
        for raw in f:
            line = raw.strip()
            if line.startswith('Time_Start,Time_Stop,Time_Mid,CO_ppm'):
                raw_cols = [c.strip() for c in line.split(',')]
                break
            header_lines.append(raw)

        # mangle duplicates exactly as before
        from collections import defaultdict
        counter = defaultdict(int)
        cols = []
        for name in raw_cols:
            if counter[name]:
                name = f"{name}_{counter[name]}"
            cols.append(name)
            counter[name] += 1

        # figure out which of our target cols actually ended up in `cols`
        usecols = [c for c in cols if c in target]

        # now read only those columns
        df = pd.read_csv(
            f,
            header=None,
            names=cols,
            usecols=usecols,
            na_values=na_values,
            comment='#'
        )
        
        df.loc[df['CO_ppm'] < 0, 'CO_ppm'] = np.nan
        
        # convert to DataFrame
        co_df = pd.DataFrame({'tmhr': df['Time_Mid']/86400*24.0, 
                               'co': df['CO_ppm']*1000.0,  # convert to ppb
                               })
        
    return header_lines, co_df


def read_ict_co2(filepath, encoding='utf-8', na_values=None):
    """
    Reads an ICT file but only loads the columns:
      Time_Start,Time_Stop,Time_Mid,CO2_ppm
      
    Returns:
      header_lines: list of strings (the metadata/header)
      df: pandas DataFrame of just those four fields
    """
    if na_values is None:
        na_values = [-9999, -8888, -7777]

    # the only columns we care about
    target = ["Time_Start", "Time_Stop", "Time_Mid", "CO2_ppm"]

    with open(filepath, 'r', encoding=encoding, errors='replace') as f:
        header_lines = []
        for raw in f:
            line = raw.strip()
            if line.startswith('Time_Start,Time_Stop,Time_Mid,CO2_ppm'):
                raw_cols = [c.strip() for c in line.split(',')]
                break
            header_lines.append(raw)

        # mangle duplicates exactly as before
        from collections import defaultdict
        counter = defaultdict(int)
        cols = []
        for name in raw_cols:
            if counter[name]:
                name = f"{name}_{counter[name]}"
            cols.append(name)
            counter[name] += 1

        # figure out which of our target cols actually ended up in `cols`
        usecols = [c for c in cols if c in target]

        # now read only those columns
        df = pd.read_csv(
            f,
            header=None,
            names=cols,
            usecols=usecols,
            na_values=na_values,
            comment='#'
        )
        
        df.loc[df['CO2_ppm'] < 0, 'CO2_ppm'] = np.nan
        
        # convert to DataFrame
        co2_df = pd.DataFrame({'tmhr': df['Time_Mid']/86400*24.0, 
                               'co2': df['CO2_ppm'], # in ppm
                               })
        
    return header_lines, co2_df


def read_ict_o3(filepath, encoding='utf-8', na_values=None):
    """
    Reads an ICT file but only loads the columns:
      Time_Start,Time_Stop,Time_Mid,O3_ppbv
      
    Returns:
      header_lines: list of strings (the metadata/header)
      df: pandas DataFrame of just those four fields
    """
    if na_values is None:
        na_values = [-9999, -8888, -7777]

    # the only columns we care about
    target = ["Time_Start", "Time_Stop", "Time_Mid", "O3_ppbv"]

    with open(filepath, 'r', encoding=encoding, errors='replace') as f:
        header_lines = []
        for raw in f:
            line = raw.strip()
            if line.startswith('Time_Start,Time_Stop,Time_Mid,O3_ppbv'):
                raw_cols = [c.strip() for c in line.split(',')]
                break
            header_lines.append(raw)

        # mangle duplicates exactly as before
        from collections import defaultdict
        counter = defaultdict(int)
        cols = []
        for name in raw_cols:
            if counter[name]:
                name = f"{name}_{counter[name]}"
            cols.append(name)
            counter[name] += 1

        # figure out which of our target cols actually ended up in `cols`
        usecols = [c for c in cols if c in target]

        # now read only those columns
        df = pd.read_csv(
            f,
            header=None,
            names=cols,
            usecols=usecols,
            na_values=na_values,
            comment='#'
        )
        
        df.loc[df['O3_ppm'] < 0, 'O3_ppm'] = np.nan
        
        # convert to DataFrame
        o3_df = pd.DataFrame({'tmhr': df['Time_Mid']/86400*24.0, 
                               'o3': df['O3_ppbv'], # in ppb
                               })
        
    return header_lines, o3_df


def ssfr_slit_convolve(wvl, flux_orig, wvl_joint):
    dwvl = wvl[1] - wvl[0]
    xx = np.linspace(-12, 12, int(24/dwvl+1))
    yy_gaussian_vis = gaussian(xx, 0, 3.8251)
    yy_gaussian_nir = gaussian(xx, 0, 4.5046)
    
    flux_conv = flux_orig.copy()
    
    flux_convolved_vis = convolve(flux_orig, yy_gaussian_vis, mode='same') / np.sum(yy_gaussian_vis)
    flux_convolved_nir = convolve(flux_orig, yy_gaussian_nir, mode='same') / np.sum(yy_gaussian_nir)
    flux_conv[wvl<=wvl_joint] = flux_convolved_vis[wvl<=950]
    flux_conv[wvl>wvl_joint] = flux_convolved_nir[wvl>950]
    
    # plt.close('all')
    # plt.figure(figsize=(10,6))
    # plt.plot(wvl, flux_orig, label='Original', color='black')
    # plt.plot(wvl, flux_conv, label='Convolved', color='red')
    # plt.xlabel('Wavelength (nm)')
    # plt.ylabel('Flux')
    # plt.title('SSFR Slit Function Convolution')
    # plt.legend()
    # plt.grid()
    # plt.show()
    # sys.exit()
    
    return flux_conv

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
        root = self.root_mac if sys.platform=="darwin" else self.root_linux
        return f"{root}/marli/ARCSIX-MARLi_P3B_{date_s}_R0.cdf"
    def kt19(self, date_s):    
        root = self.root_mac if sys.platform=="darwin" else self.root_linux
        return f"{root}/kt19/ARCSIX-MetNav-KT19-10Hz_P3B_{date_s}_R0.ict"
    def sat_nc(self, date_s, raw):  # choose root by platform
        root = self.root_mac if sys.platform=="darwin" else self.root_linux
        return f"{root}/sat-data/{date_s}/{raw}"
    
    
# --- Helpers ----------------------------------------------------------------

def load_h5(path):
    if not os.path.exists(path): raise FileNotFoundError(path)
    return er3t.util.load_h5(str(path))

def parse_sat(path):
    with h5py.File(str(path),"r") as f:
        desc = f["sat/jday"].attrs["description"].split("\n")
    names, tmhrs, files = [], [], []
    for raw in desc:
        base = Path(raw).stem.replace("CLDPROP_L2_","")
        hh, mm = float(base.split(".")[2][:4].rjust(4,"0")[:2]), float(base.split(".")[2][2:4])
        print("hh, mm:", hh, mm)
        tm = hh + mm/60.0
        names.append(base.split(".")[0].replace("_"," "))
        tmhrs.append(tm)
        files.append(Path(raw).name)
    return names, np.array(tmhrs), files

def nearest_indices(t_hsk, mask, times):
    # vectorized nearest‐index lookup per leg
    return np.argmin(np.abs(times[:,None] - t_hsk[mask][None,:]), axis=0)

def closest_indices(available: np.ndarray, targets: np.ndarray):
    # vectorized closest-index
    return np.argmin(np.abs(available[:,None] - targets[None,:]), axis=0)

def dropsonde_time_loc_list(dir_dropsonde=f'data/dropsonde'):
    """
    Get the dropsonde time list from the dropsonde directory.
    
    Parameters
    ----------
    dir_dropsonde : str
        The directory of dropsonde files.
        
    Returns
    -------
    time_list : list of datetime
        The list of dropsonde times.
    file_list : list of str
        The list of dropsonde file names.
    """
    file_list = sorted(glob.glob(os.path.join(dir_dropsonde, '*.ict')))
    # glob daughter directories 
    dir_list = sorted([d for d in glob.glob(os.path.join(dir_dropsonde, '*')) if os.path.isdir(d)])
    for d in dir_list:
        file_list.extend(sorted(glob.glob(os.path.join(d, '*.ict'))))
    date_list = []
    tmhr_list = []
    lon_list = []
    lat_list = []
    for f in file_list:
        fname = os.path.basename(f)
        # Example filename: ARCSIX-AVAPS_G3_20240531_R0/ARCSIX-AVAPS_G3_20240531142150_R0.ict
        date_str = fname.split('_')[2]  # '20240531142150'
        date_time = datetime.datetime.strptime(date_str, '%Y%m%d%H%M%S')
        date_list.append(date_time.date())
        
        tmhr_list.append(date_time.hour + date_time.minute/60 + date_time.second/3600)
        
        head, data_dropsonde = read_ict_dropsonde(f, encoding='utf-8', na_values=[-9999999, -777, -888])
        lon_list.append(np.mean(data_dropsonde['lon_all']))
        lat_list.append(np.mean(data_dropsonde['lat_all']))
    return np.array(file_list), np.array(date_list), np.array(tmhr_list), np.array(lon_list), np.array(lat_list)

if __name__ == '__main__':


    pass
