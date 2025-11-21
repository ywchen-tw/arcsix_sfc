import os
import sys
import glob
import copy
import time
from collections import OrderedDict
import datetime
import multiprocessing as mp
import pickle
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


def prepare_atmospheric_profile(fdir_data, date_s, case_tag, ileg, date, time_start, time_end,
                                alt_avg, data_dropsonde,
                                cld_leg, levels=None,
                                mod_extent=[-60.0, -80.0, 82.4, 84.6],
                                zpt_filedir='./data/atmospheric_profiles'
                                ):
    
    from er3t.util.modis import get_filename_tag
    from util.modis07_download import modis_download
    
    
    mod_extent = [np.float64(i) for i in mod_extent]
    
    aqua_time_tag = get_filename_tag(date, np.array(mod_extent[:2]), np.array(mod_extent[2:]), satID='aqua')
    terra_time_tag = get_filename_tag(date, np.array(mod_extent[:2]), np.array(mod_extent[2:]), satID='terra')
    
    sat_time_tags, sat_time_tags_int, sat_id_list = [], [], []
    if aqua_time_tag is not None:
        sat_time_tags.extend(aqua_time_tag)
        sat_time_tags_int.extend([int(t.split('.')[1]) for t in aqua_time_tag])
        sat_id_list.extend(['aqua']*len(aqua_time_tag))
    if terra_time_tag is not None:
        sat_time_tags.extend(terra_time_tag)
        sat_time_tags_int.extend([int(t.split('.')[1]) for t in terra_time_tag])
        sat_id_list.extend(['terra']*len(terra_time_tag))
    
    time_s = f'{np.floor(time_start):2.0f}{int((time_start - np.floor(time_start))*60):02d}'
    
    # find the closest overpass time
    time_diffs = [abs(int(t) - int(time_s)) for t in sat_time_tags_int]
    if len(time_diffs) == 0:
        raise ValueError("No MODIS overpass found for the given date and extent")
    min_ind = np.argmin(time_diffs)
    satID = sat_id_list[min_ind]
    sat_time_tag_final = sat_time_tags[min_ind]
    
    # print("Available MODIS overpass times and IDs:", sat_time_tags, sat_id_list)
    # print(f"Selected MODIS satellite: {satID} {sat_time_tag_final} for atmospheric profile generation")
    
    sat0 = modis_download(date=date, 
                          satID=satID,
                              fdir_out='../data/sat-data', 
                              fdir_pre_data='../data/sat-data',
                              extent=mod_extent,
                              extent_analysis=mod_extent,
                              filename_tag=sat_time_tag_final,
                              fname=f'modis_{date_s}_{time_s}.pk', overwrite=False)
    
    modis_07_file = sat0.fnames['mod_07']
    
    # print("Using MODIS 07 file:", modis_07_file)
    
    zpt_filename = f'zpt_{date_s}_{case_tag}_leg_{ileg}.h5'
                
    fname_atm = f'modis_dropsonde_atm_{date_s}_{case_tag}_leg_{ileg}.pk'
    
    status, ws10m = er3t.pre.atm.create_modis_dropsonde_atm(o2mix=0.20935, output_dir=zpt_filedir, output=zpt_filename, 
                                            fname_mod07=modis_07_file, dropsonde_df=data_dropsonde,
                                            levels=levels,
                                            extent=mod_extent, new_h_edge=None,sfc_T_set=None, sfc_h_to_zero=True,)
    
    fname_insitu = f'{fdir_data}/zpt/{date_s}/{date_s}_gases_profiles.csv'
    if not os.path.exists(fname_insitu):
        fname_insitu = None
        
    atm0      = er3t.pre.atm.modis_dropsonde_arcsix_atmmod(zpt_file=f'{zpt_filedir}/{zpt_filename}',
                        fname=fname_atm, 
                        fname_co2_clim=f'{fdir_data}/climatology/cams73_latest_co2_conc_surface_inst_2020.nc',
                        fname_ch4_clim=f'{fdir_data}/climatology/cams_ch4_202005-202008.nc',
                        fname_o3_clim=f'{fdir_data}/climatology/ozone_merra2_202405_202408.h5',
                        fname_insitu=fname_insitu,
                        marli_h=cld_leg['marli_h'], marli_wvmr=cld_leg['marli_wvmr'],
                        date=date, extent=mod_extent,
                        overwrite=True)
    
    if np.any(np.isnan(atm0.lev["pressure"]["data"])):
        raise ValueError("NaN values found in pressure profile, please check the dropsonde data and MODIS 07 data coverage")

    # write out the atmospheric profile in ascii format
    with open(os.path.join(zpt_filedir, f'atm_profiles_{date_s}_{case_tag}_{time_start:.2f}_{time_end:.2f}_{alt_avg:.2f}km.dat'), 'w') as f:
        header = ('# Combined atmospheric profile\n'
                '#      z(km)      p(mb)        T(K)    air(cm-3)    o3(cm-3)     o2(cm-3)    h2o(cm-3)    co2(cm-3)     no2(cm-3)\n'
                )
        # Build all profile lines in one go.
        lines = [
                f'{atm0.lev["altitude"]["data"][i]:11.3f} {atm0.lev["pressure"]["data"][i]:11.5f} {atm0.lev["temperature"]["data"][i]:11.3f} '
                f'{atm0.lev["air"]["data"][i]:12.6e} {atm0.lev["o3"]["data"][i]:12.6e} {atm0.lev["o2"]["data"][i]:12.6e} '
                f'{atm0.lev["h2o"]["data"][i]:12.6e} {atm0.lev["co2"]["data"][i]:12.6e} {atm0.lev["no2"]["data"][i]:12.6e}'
                for i in range(len(atm0.lev['altitude']['data']))[::-1]
                ]
        f.write(header + "\n".join(lines))
    
    with open(f'{zpt_filedir}/ch4_profiles_{date_s}_{case_tag}_{time_start:.2f}_{time_end:.2f}_{alt_avg:.2f}km.dat', 'w') as f:  
        header = ('# Combined atmospheric profile for ch4 only\n'
                '#      z(km)      ch4(cm-3)\n'
                )
        lines = [
                f'{atm0.lev["altitude"]["data"][i]:11.3f} {atm0.lev["ch4"]["data"][i]:12.6e}'
                for i in range(len(atm0.lev['altitude']['data']))[::-1]
                ]
        f.write(header + "\n".join(lines))
        
    return None





if __name__ == '__main__':


    pass
