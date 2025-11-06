"""
by Hong Chen (hong.chen@lasp.colorado.edu)

This code serves as an example code to reproduce 3D irradiance simulation for App. 3 in Chen et al. (2022).
Special note: due to large data volume, only partial flight track simulation is provided for illustration purpose.

The processes include:
    1) `main_run()`: pre-process aircraft and satellite data and run simulations
        a) partition flight track into mini flight track segments and collocate satellite imagery data
        b) run simulations based on satellite imagery cloud retrievals
            i) 3D mode
            ii) IPA mode

    2) `main_post()`: post-process data
        a) extract radiance observations from pre-processed data
        b) extract radiance simulations of EaR3T
        c) plot

This code has been tested under:
    1) Linux on 2023-06-27 by Hong Chen
      Operating System: Red Hat Enterprise Linux
           CPE OS Name: cpe:/o:redhat:enterprise_linux:7.7:GA:workstation
                Kernel: Linux 3.10.0-1062.9.1.el7.x86_64
          Architecture: x86-64
"""

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
from scipy import interpolate
# mpl.use('Agg')


import er3t


_mission_      = 'arcsix'


legend_size = 12
label_size = 14


_h = 6.626e-34
_c = 3.0e+8
_k = 1.38e-23

def planck(wvl_nm, T):
    """Calculate the Planck function for a given wavelength and temperature.
    Args:
        wav (float or float array): Wavelength in nm.
        T (float or float array): Temperature in Kelvin.
    Returns:
        float or float array: Irradiance in W/m^2/m.
    """
    wvl_m = wvl_nm * 1e-9  # convert nm to m
    a = 2.0*np.pi*_h*_c**2
    b = _h*_c/(wvl_m*_k*T)
    irradiance_per_m = a/ ( (wvl_m**5) * (np.exp(b) - 1.0) )
    irradiance_per_nm = irradiance_per_m * 1e-9  # convert from m to nm
    return irradiance_per_nm


def plot_cld_var(date_list, sat_list, manual_list, 
                 xlabel="COT", ylabel="Density",
                 title="Cloud Optical Thickness (COT)",
                 var_name='cot', fig_tag='test',
                 label_size=12, legend_size=8):
    fig = plt.figure(figsize=(5, 2.9))
    ax1 = fig.add_subplot(111)
    
    color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
    for ind, j in enumerate(set(date_list)):
        mask = np.array(date_list) == j
        print(f"Processing date: {j}, mask length: {np.sum(mask)}")
            
        date_label = j
        
        ax1.hist(np.array(sat_list)[mask], bins=25, alpha=0.35, label=f'satellite ({date_label})', density=True, color=color_list[ind],)
        ax1.axvline(np.nanmean(np.array(manual_list)[mask]), linestyle='--', linewidth=2,
                    label=f'In situ ({date_label})', 
                    color=color_list[ind])
    xmin, xmax = ax1.get_xlim()
    ax1.set_xlim(0, xmax)
    ax1.set_title(title, fontsize=label_size+2)
    ax1.set_xlabel(xlabel, fontsize=label_size)
    ax1.set_ylabel(ylabel, fontsize=label_size)
    ax1.legend(fontsize=legend_size)
    ax1.tick_params('both', labelsize=label_size-2)
    # fig.suptitle(f"Clouds on {date}", fontsize=label_size+2, y=0.98)
    fig.tight_layout()
    fig.savefig(f"fig/clouds/clouds_{var_name}_compare_{fig_tag}.png", dpi=300, bbox_inches='tight')


def flt_trk_cre_2lay_sw(dates=['20240611', '20240613'],
                          case_tag=['default', 'default'],
                          separate_heights=[1000, 2000],
                          fig_tag='test',
                          overwrite=False
                            ):


    manual_cloud_tag = 'manual_cloud'
    sat_cloud_tag = 'sat_cloud'
    clear_tag = 'clear'
    
    lrt_fname = 'flt_trk_lrt_para-lrt-{}-{}-{}.h5'
    
    seg_list = []
    alt_list = []
    date_list = []
    sza_list = []
    # ssfr_zen_flux_list = []
    # ssfr_nad_flux_list = []
    # hsr1_total_flux_list = []
    # hsr1_diff_flux_list = []
    # bbr_up_flux_list = []
    # bbr_down_flux_list = []
    # bbr_sky_T_list = []
    # kt19_T_list = []
    sat_cot_list = []
    sat_cth_list = []
    sat_cbh_list = []
    sat_cwp_list = []
    sat_cer_list = []
    sat_f_down_list = []
    sat_f_down_dir_list = []
    sat_f_down_diff_list = []
    sat_f_up_list = []
    manual_cot_list = []
    manual_cth_list = []
    manual_cbh_list = []
    manual_cwp_list = []
    manual_cer_list = []
    manual_f_down_list = []
    manual_f_down_dir_list = []
    manual_f_down_diff_list = []
    manual_f_up_list = []
    clear_f_down_list = []
    clear_f_down_dir_list = []
    clear_f_down_diff_list = []
    clear_f_up_list = []
    
    dwvl_nm = 1
    
    for i, (date, tag, sep_H) in enumerate(zip(dates, case_tag, separate_heights)):
        print(f"Processing date: {date}, case_tag: {tag}")
        
        # process sat cloud data
        lrt_fname_full = lrt_fname.format(date, tag, sat_cloud_tag)
        
        if not os.path.exists(lrt_fname_full) or overwrite:
            raise FileExistsError(f"File {lrt_fname_full} does not exist...")
            # Process the data and create the file

        
        with h5py.File(lrt_fname_full, 'r') as f:
            alt = f['alt'][...]
            sza = f['sza'][...]
            cth = f['cth'][...]
            cbh = f['cbh'][...]
            cot = f['cot'][...]
            cwp = f['cwp'][...]*1000  # convert to g m^-2
            cer = f['cer'][...]
            f_down = f['f_down'][...][:, :, 0] # select at sfc
            f_down_dir = f['f_down_dir'][...][:, :, 0]
            f_down_diff = f['f_down_diff'][...][:, :, 0]
            f_up = f['f_up'][...][:, :, 0]
            
            print("lrt_fname_full:", lrt_fname_full)
            print("alt shape:", alt.shape)
            print("f_down shape:", f_down.shape)
            print("f_down[:, 0] >0:", np.sum(f_down[:, 0] > 0))
        
            # ssfr_zen = f['ssfr_zen'][...]
            # ssfr_nad = f['ssfr_nad'][...]
            # ssfr_zen_wvl = f['ssfr_zen_wvl'][...]
            # ssfr_nad_wvl = f['ssfr_nad_wvl'][...]
            # ssfr_toa0 = f['ssfr_toa0'][...]
            
            # hsr1_wvl = f['hsr1_wvl'][...]
            # hsr1_toa0 = f['hsr1_toa0'][...]
            # hsr1_total = f['hsr1_total'][...]
            # hsr1_diff = f['hsr1_diff'][...]
            
            # bbr_up = f['bbr_up'][...]
            # bbr_down = f['bbr_down'][...]
            # bbr_sky_T = f['bbr_sky_T'][...] + 273.15  # convert to Kelvin
            # kt19_T = f['kt19'][...] + 273.15  # convert to Kelvin
        
        date_repeat = np.repeat(date, alt.shape[0])
        date_list.extend(date_repeat)
        alt_list.extend(alt)
        sza_list.extend(sza)
        
        
        seg = np.zeros_like(alt, dtype=int)
        seg[alt < sep_H] = i+1  # low altitude segment
        seg[alt >= sep_H] = (i+1)*1000+1  # high altitude segment
        seg_list.extend(seg)
        
        sat_f_down_list.extend(np.sum(f_down*dwvl_nm, axis=1))  # sum over wavelengths
        sat_f_down_dir_list.extend(np.sum(f_down_dir*dwvl_nm, axis=1))
        sat_f_down_diff_list.extend(np.sum(f_down_diff*dwvl_nm, axis=1))
        sat_f_up_list.extend(np.sum(f_up*dwvl_nm, axis=1))
        
        
        cth = np.where(cth < 0, np.nan, cth)  # set negative values to NaN
        sat_cot_list.extend(cot)
        sat_cth_list.extend(cth)
        sat_cbh_list.extend(cbh)
        sat_cwp_list.extend(cwp)
        sat_cer_list.extend(cer)
        
        # process manual cloud data
        lrt_fname_full = lrt_fname.format(date, tag, manual_cloud_tag)
        if not os.path.exists(lrt_fname_full) or overwrite:
            raise FileExistsError(f"File {lrt_fname_full} does not exist...")
        
        with h5py.File(lrt_fname_full, 'r') as f:
            cth = f['cth'][...]  # convert km
            cbh = f['cbh'][...]  # convert km
            cot = f['cot'][...]
            cwp = f['cwp'][...]*1000  # convert to g m^-2
            cer = f['cer'][...]
            f_down = f['f_down'][...][:, :, 0]
            f_down_dir = f['f_down_dir'][...][:, :, 0]
            f_down_diff = f['f_down_diff'][...][:, :, 0]
            f_up = f['f_up'][...][:, :, 0]
            
        manual_f_down_list.extend(np.sum(f_down*dwvl_nm, axis=1))  # sum over wavelengths
        manual_f_down_dir_list.extend(np.sum(f_down_dir*dwvl_nm, axis=1))
        manual_f_down_diff_list.extend(np.sum(f_down_diff*dwvl_nm, axis=1))
        manual_f_up_list.extend(np.sum(f_up*dwvl_nm, axis=1))   
        
        cth = np.where(cth < 0, np.nan, cth)  # set negative values to NaN
        manual_cot_list.extend(cot)
        manual_cth_list.extend(cth)
        manual_cbh_list.extend(cbh)
        manual_cwp_list.extend(cwp)
        manual_cer_list.extend(cer)
        
        # process clear sky data
        lrt_fname_full = lrt_fname.format(date, tag, clear_tag)
        if not os.path.exists(lrt_fname_full) or overwrite:
            raise FileExistsError(f"File {lrt_fname_full} does not exist...")
        with h5py.File(lrt_fname_full, 'r') as f:
            f_down = f['f_down'][...][:, :, 0]
            f_down_dir = f['f_down_dir'][...][:, :, 0]
            f_down_diff = f['f_down_diff'][...][:, :, 0]
            f_up = f['f_up'][...][:, :, 0]
        
        clear_f_down_list.extend(np.sum(f_down*dwvl_nm, axis=1))  # sum over wavelengths
        clear_f_down_dir_list.extend(np.sum(f_down_dir*dwvl_nm, axis=1))
        clear_f_down_diff_list.extend(np.sum(f_down_diff*dwvl_nm, axis=1))
        clear_f_up_list.extend(np.sum(f_up*dwvl_nm, axis=1))
        
    
    # convert lists to pandas dataframes
    
    # print("SW sat cth list:", (sat_cth_list))
    # print("SW manual cth list:", (manual_cth_list))
    # sys.exit()
    
    df = pd.DataFrame({
        'date': date_list,
        'alt': alt_list,
        'sza': sza_list,
        'seg': seg_list,
        'sat_cot': sat_cot_list,
        'sat_cth': sat_cth_list,
        'sat_cbh': sat_cbh_list,
        'sat_cwp': sat_cwp_list,
        'sat_cer': sat_cer_list,
        'sat_f_down': sat_f_down_list,
        'sat_f_down_dir': sat_f_down_dir_list,
        'sat_f_down_diff': sat_f_down_diff_list,
        'sat_f_up': sat_f_up_list,
        'manual_cot': manual_cot_list,
        'manual_cth': manual_cth_list,
        'manual_bcb': manual_cbh_list,
        'manual_cwp': manual_cwp_list,
        'manual_cer': manual_cer_list,
        'manual_f_down': manual_f_down_list,
        'manual_f_down_dir': manual_f_down_dir_list,
        'manual_f_down_diff': manual_f_down_diff_list,
        'manual_f_up': manual_f_up_list,
        'clear_f_down': clear_f_down_list,
        'clear_f_down_dir': clear_f_down_dir_list,
        'clear_f_down_diff': clear_f_down_diff_list,
        'clear_f_up': clear_f_up_list
    })
    
    

    sat_f_down_nan_mask = np.isnan(sat_f_down_list)
    
    for col in ['sat_cot', 'sat_cth', 'sat_cwp', 'sat_cer',
                'manual_cot', 'manual_cth', 'manual_cwp', 'manual_cer',]:
        df.loc[df[col]<0, col] = np.nan  # set negative values to NaN
    
    df = df[~sat_f_down_nan_mask]  # remove rows with NaN in sat_cot
    

    F_sat_cloud = df['sat_f_down'] - df['sat_f_up']
    F_manual_cloud = df['manual_f_down'] - df['manual_f_up']
    F_clear = df['clear_f_down'] - df['clear_f_up']
    
    F_cre_sat_cloud = np.array(F_sat_cloud - F_clear)
    F_cre_manual_cloud = np.array(F_manual_cloud - F_clear)
    
    # print("F_cre_sat_cloud:", F_cre_sat_cloud)
    # print("F_cre_manual_cloud:", F_cre_manual_cloud)
    
    # sys.exit()
    
    F_cre_sat_list = []
    F_cre_manual_list = []
    F_cre_sat_postion_list = []
    F_cre_manual_postion_list = []
    date_postion_list = []
    date_label_list = []
    
    date_sat_cer_list = []
    date_sat_cwp_list = []
    date_sat_cot_list = []
    date_sat_cth_list = []
    date_sat_cbh_list = []
    date_manual_cer_list = []
    date_manual_cwp_list = []
    date_manual_cot_list = []
    date_manual_cth_list = []
    date_manual_cbh_list = []
    
    # plot the results
    fig = plt.figure(figsize=(5, 2.9))
    ax1 = fig.add_subplot(111)
    
    color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
    for ind, j in enumerate(sorted(set(date_list))):
        mask = np.array(df['date']) == j
        print(f"Processing date: {j}, mask length: {np.sum(mask)}")

        mm = np.int(j[4:6])
        dd = np.int(j[6:8])
        date_label = f"{mm:d}/{dd:d}"

        print('date_label:', date_label)
        print("F_cre_sat_cloud sw shape:", np.shape(F_cre_sat_cloud[mask]))
        
        F_cre_sat_list.append(F_cre_sat_cloud[mask])
        F_cre_manual_list.append(F_cre_manual_cloud[mask])
        
        F_cre_sat_postion_list.append(ind*2-0.25)
        F_cre_manual_postion_list.append(ind*2+0.25)
        date_postion_list.append(ind*2)
        date_label_list.append(date_label)
        
        date_sat_cer_list.append(np.array(df['sat_cer'][mask]))
        date_sat_cwp_list.append(np.array(df['sat_cwp'][mask]))
        date_sat_cot_list.append(np.array(df['sat_cot'][mask]))
        date_sat_cth_list.append(np.array(df['sat_cth'][mask]))
        date_sat_cbh_list.append(np.array(df['sat_cbh'][mask]))
        date_manual_cer_list.append(np.array(df['manual_cer'][mask]))
        date_manual_cwp_list.append(np.array(df['manual_cwp'][mask]))
        date_manual_cot_list.append(np.array(df['manual_cot'][mask]))
        date_manual_cth_list.append(np.array(df['manual_cth'][mask]))
        date_manual_cbh_list.append(np.array(df['manual_bcb'][mask]))
    
    bp1 = ax1.boxplot(F_cre_sat_list, positions=F_cre_sat_postion_list, widths=0.25,
                        patch_artist=True, boxprops=dict(facecolor='tab:red', color='tab:red'),
                        medianprops=dict(color='black'), showfliers=False, showmeans=True,
                        meanline=True, meanprops=dict(color='black', linestyle='--'))
    bp2 = ax1.boxplot(F_cre_manual_list, positions=F_cre_manual_postion_list, widths=0.25,
                        patch_artist=True, boxprops=dict(facecolor='tab:green', color='tab:blue'),
                        medianprops=dict(color='black'), showfliers=False, showmeans=True,
                        meanline=True, meanprops=dict(color='black', linestyle='--'))
        
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        for j in bp1[element]:
            j.set(color='r',)# linewidth=1.5)
        for j in bp2[element]:
            j.set(color='g',)# linewidth=1.5)
        for patch in bp1['boxes']:
            patch.set(facecolor='lightsalmon')  # Set the color of the box
        for patch in bp2['boxes']:
            patch.set(facecolor='palegreen')  # Set the color of the box
    # ax1.set_ylim(-122, 122)
    ax1.axhline(0, color='black', linestyle='--', linewidth=0.5)
    
    # ax1.legend([bp1["boxes"][0], bp2["boxes"][0]], ['satellite cloud', 'measured cloud'], loc='best', fontsize=legend_size)
    
    date_postion_list = np.array(date_postion_list)
    ax1.set_xticks(date_postion_list, date_label_list,)# rotation=45, ha='right')
    ax1.set_ylabel("SW CRE\n(W m$^{-2}$)", fontsize=label_size)
    ax1.set_xlabel("Date", fontsize=label_size)
    
    ax1.tick_params('both', labelsize=label_size-2)
    # fig.suptitle(f"Clouds on {date}", fontsize=label_size+2, y=0.98)
    fig.tight_layout()
    fig.savefig(f"fig/clouds/CRE_compare_SW_{fig_tag}.png", dpi=300, bbox_inches='tight')
    
    
    results = {'date_label_list': date_label_list,
               'F_cre_sat_list': F_cre_sat_list,
               'F_cre_manual_list': F_cre_manual_list,
               'F_cre_sat_postion_list': F_cre_sat_postion_list,
               'F_cre_manual_postion_list': F_cre_manual_postion_list,
               'date_postion_list': date_postion_list,
               'sat_cer_list': date_sat_cer_list,
               'sat_cwp_list': date_sat_cwp_list,
               'sat_cot_list': date_sat_cot_list,
               'sat_cth_list': date_sat_cth_list,
               'sat_cbh_list': date_sat_cbh_list,
               'manual_cer_list': date_manual_cer_list,
               'manual_cwp_list': date_manual_cwp_list,
               'manual_cot_list': date_manual_cot_list,
                'manual_cth_list': date_manual_cth_list,
                'manual_cbh_list': date_manual_cbh_list,
               }
    
    return results
 

def flt_trk_cre_2lay_lw(dates=['20240611', '20240613'],
                          case_tag=['default', 'default'],
                          separate_heights=[1000, 2000],
                          fig_tag='test',
                          overwrite=False
                            ):


    manual_cloud_tag = 'manual_cloud'
    sat_cloud_tag = 'sat_cloud'
    clear_tag = 'clear'
    
    lrt_fname = 'flt_trk_lrt_para-lrt-{}-{}-{}-lw.h5'
    
    seg_list = []
    alt_list = []
    date_list = []
    sza_list = []
    # ssfr_zen_flux_list = []
    # ssfr_nad_flux_list = []
    # hsr1_total_flux_list = []
    # hsr1_diff_flux_list = []
    # bbr_up_flux_list = []
    # bbr_down_flux_list = []
    # bbr_sky_T_list = []
    # kt19_T_list = []
    sat_cot_list = []
    sat_cth_list = []
    sat_cbh_list = []
    sat_cwp_list = []
    sat_cer_list = []
    sat_f_down_list = []
    sat_f_down_dir_list = []
    sat_f_down_diff_list = []
    sat_f_up_list = []
    manual_cot_list = []
    manual_cth_list = []
    manual_cbh_list = []
    manual_cwp_list = []
    manual_cer_list = []
    manual_f_down_list = []
    manual_f_down_dir_list = []
    manual_f_down_diff_list = []
    manual_f_up_list = []
    clear_f_down_list = []
    clear_f_down_dir_list = []
    clear_f_down_diff_list = []
    clear_f_up_list = []
    
    dwvl_nm = 1
    
    for i, (date, tag, sep_H) in enumerate(zip(dates, case_tag, separate_heights)):
        print(f"Processing date: {date}, case_tag: {tag}")
        
        # process sat cloud data
        lrt_fname_full = lrt_fname.format(date, tag, sat_cloud_tag)
        
        if not os.path.exists(lrt_fname_full) or overwrite:
            raise FileExistsError(f"File {lrt_fname_full} does not exist...")
            # Process the data and create the file

        
        with h5py.File(lrt_fname_full, 'r') as f:
            alt = f['alt'][...]
            sza = f['sza'][...]
            cth = f['cth'][...]
            cbh = f['cbh'][...]
            cot = f['cot'][...]
            cwp = f['cwp'][...]*1000  # convert to g m^-2
            cer = f['cer'][...]
            f_down = f['f_down'][...][:, :, 0]*1000 # select at TOA
            f_down_dir = f['f_down_dir'][...][:, :, 0]*1000
            f_down_diff = f['f_down_diff'][...][:, :, 0]*1000
            f_up = f['f_up'][...][:, :, 0]*1000
        
            # ssfr_zen = f['ssfr_zen'][...]
            # ssfr_nad = f['ssfr_nad'][...]
            # ssfr_zen_wvl = f['ssfr_zen_wvl'][...]
            # ssfr_nad_wvl = f['ssfr_nad_wvl'][...]
            # ssfr_toa0 = f['ssfr_toa0'][...]
            
            # hsr1_wvl = f['hsr1_wvl'][...]
            # hsr1_toa0 = f['hsr1_toa0'][...]
            # hsr1_total = f['hsr1_total'][...]
            # hsr1_diff = f['hsr1_diff'][...]
            
            # bbr_up = f['bbr_up'][...]
            # bbr_down = f['bbr_down'][...]
            # bbr_sky_T = f['bbr_sky_T'][...] + 273.15  # convert to Kelvin
            # kt19_T = f['kt19'][...] + 273.15  # convert to Kelvin
        
        date_repeat = np.repeat(date, alt.shape[0])
        date_list.extend(date_repeat)
        alt_list.extend(alt)
        sza_list.extend(sza)
        
        
        seg = np.zeros_like(alt, dtype=int)
        seg[alt < sep_H] = i+1  # low altitude segment
        seg[alt >= sep_H] = (i+1)*1000+1  # high altitude segment
        seg_list.extend(seg)
        
        sat_f_down_list.extend(np.sum(f_down*dwvl_nm, axis=1))  # sum over wavelengths
        sat_f_down_dir_list.extend(np.sum(f_down_dir*dwvl_nm, axis=1))
        sat_f_down_diff_list.extend(np.sum(f_down_diff*dwvl_nm, axis=1))
        sat_f_up_list.extend(np.sum(f_up*dwvl_nm, axis=1))
        
        
        cth = np.where(cth < 0, np.nan, cth)  # set negative values to NaN
        sat_cot_list.extend(cot)
        sat_cth_list.extend(cth)
        sat_cbh_list.extend(cbh)
        sat_cwp_list.extend(cwp)
        sat_cer_list.extend(cer)
        
        # process manual cloud data
        lrt_fname_full = lrt_fname.format(date, tag, manual_cloud_tag)
        if not os.path.exists(lrt_fname_full) or overwrite:
            raise FileExistsError(f"File {lrt_fname_full} does not exist...")
        
        with h5py.File(lrt_fname_full, 'r') as f:
            cth = f['cth'][...]  # convert km
            cbh = f['cbh'][...]  # convert km
            cot = f['cot'][...]
            cwp = f['cwp'][...]*1000  # convert to g m^-2
            cer = f['cer'][...]
            f_down = f['f_down'][...][:, :, 0]*1000
            f_down_dir = f['f_down_dir'][...][:, :, 0]*1000
            f_down_diff = f['f_down_diff'][...][:, :, 0]*1000
            f_up = f['f_up'][...][:, :, 0]*1000
            
        manual_f_down_list.extend(np.sum(f_down*dwvl_nm, axis=1))  # sum over wavelengths
        manual_f_down_dir_list.extend(np.sum(f_down_dir*dwvl_nm, axis=1))
        manual_f_down_diff_list.extend(np.sum(f_down_diff*dwvl_nm, axis=1))
        manual_f_up_list.extend(np.sum(f_up*dwvl_nm, axis=1))   
        
        cth = np.where(cth < 0, np.nan, cth)  # set negative values to NaN
        manual_cot_list.extend(cot)
        manual_cth_list.extend(cth)
        manual_cbh_list.extend(cbh)
        manual_cwp_list.extend(cwp)
        manual_cer_list.extend(cer)
        
        # process clear sky data
        lrt_fname_full = lrt_fname.format(date, tag, clear_tag)
        if not os.path.exists(lrt_fname_full) or overwrite:
            raise FileExistsError(f"File {lrt_fname_full} does not exist...")
        with h5py.File(lrt_fname_full, 'r') as f:
            f_down = f['f_down'][...][:, :, 0]*1000
            f_down_dir = f['f_down_dir'][...][:, :, 0]*1000
            f_down_diff = f['f_down_diff'][...][:, :, 0]*1000
            f_up = f['f_up'][...][:, :, 0]*1000
        
        clear_f_down_list.extend(np.sum(f_down*dwvl_nm, axis=1))  # sum over wavelengths
        clear_f_down_dir_list.extend(np.sum(f_down_dir*dwvl_nm, axis=1))
        clear_f_down_diff_list.extend(np.sum(f_down_diff*dwvl_nm, axis=1))
        clear_f_up_list.extend(np.sum(f_up*dwvl_nm, axis=1))
        
    
    # convert lists to pandas dataframes
    
    df = pd.DataFrame({
        'date': date_list,
        'alt': alt_list,
        'sza': sza_list,
        'seg': seg_list,
        'sat_cot': sat_cot_list,
        'sat_cth': sat_cth_list,
        'sat_cbh': sat_cbh_list,
        'sat_cwp': sat_cwp_list,
        'sat_cer': sat_cer_list,
        'sat_f_down': sat_f_down_list,
        'sat_f_down_dir': sat_f_down_dir_list,
        'sat_f_down_diff': sat_f_down_diff_list,
        'sat_f_up': sat_f_up_list,
        'manual_cot': manual_cot_list,
        'manual_cth': manual_cth_list,
        'manual_cbh': manual_cbh_list,
        'manual_cwp': manual_cwp_list,
        'manual_cer': manual_cer_list,
        'manual_f_down': manual_f_down_list,
        'manual_f_down_dir': manual_f_down_dir_list,
        'manual_f_down_diff': manual_f_down_diff_list,
        'manual_f_up': manual_f_up_list,
        'clear_f_down': clear_f_down_list,
        'clear_f_down_dir': clear_f_down_dir_list,
        'clear_f_down_diff': clear_f_down_diff_list,
        'clear_f_up': clear_f_up_list
    })
    
    sat_f_up_nan_mask = np.isnan(sat_f_up_list)
    
    for col in ['sat_cot', 'sat_cth', 'sat_cwp', 'sat_cer',
                'manual_cot', 'manual_cth', 'manual_cwp', 'manual_cer',]:
        df.loc[df[col]<0, col] = np.nan  # set negative values to NaN
    
    df = df[~sat_f_up_nan_mask]  # remove rows with NaN in sat_cot
    
    
    F_sat_cloud = df['sat_f_down'] - df['sat_f_up']
    F_manual_cloud = df['manual_f_down'] - df['manual_f_up']
    F_clear = df['clear_f_down'] - df['clear_f_up']
    
    F_cre_sat_cloud = np.array(F_sat_cloud - F_clear)
    F_cre_manual_cloud = np.array(F_manual_cloud - F_clear)
    
    F_cre_sat_list = []
    F_cre_manual_list = []
    F_cre_sat_postion_list = []
    F_cre_manual_postion_list = []
    date_postion_list = []
    date_label_list = []
    date_sat_cer_list = []
    date_sat_cwp_list = []
    date_sat_cot_list = []
    date_sat_cth_list = []
    date_sat_cbh_list = []
    date_manual_cer_list = []
    date_manual_cwp_list = []
    date_manual_cot_list = []
    date_manual_cth_list = []
    date_manual_cbh_list = []
    
    # plot the results
    fig = plt.figure(figsize=(5, 2.9))
    ax1 = fig.add_subplot(111)
    
    color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
    for ind, j in enumerate(sorted(set(date_list))):
        mask = np.array(df['date']) == j
        print(f"Processing date: {j}, mask length: {np.sum(mask)}")

        mm = np.int(j[4:6])
        dd = np.int(j[6:8])
        date_label = f"{mm:d}/{dd:d}"
        
        F_cre_sat_list.append(F_cre_sat_cloud[mask])
        F_cre_manual_list.append(F_cre_manual_cloud[mask])
        
        print("LW")
        print("date:", j)
        print("F_cre_sat_cloud shape:", np.shape(F_cre_sat_cloud[mask]))
        # print("CRE sat:", F_cre_sat_cloud[mask])
        # print("CRE manual:", F_cre_manual_cloud[mask])
        
        F_cre_sat_postion_list.append(ind*2-0.25)
        F_cre_manual_postion_list.append(ind*2+0.25)
        date_postion_list.append(ind*2)
        date_label_list.append(date_label)
        
        date_sat_cer_list.append(np.array(df['sat_cer'][mask]))
        date_sat_cwp_list.append(np.array(df['sat_cwp'][mask]))
        date_sat_cot_list.append(np.array(df['sat_cot'][mask]))
        date_sat_cth_list.append(np.array(df['sat_cth'][mask]))
        date_sat_cbh_list.append(np.array(df['sat_cbh'][mask]))
        date_manual_cer_list.append(np.array(df['manual_cer'][mask]))
        date_manual_cwp_list.append(np.array(df['manual_cwp'][mask]))
        date_manual_cot_list.append(np.array(df['manual_cot'][mask]))
        date_manual_cth_list.append(np.array(df['manual_cth'][mask]))
        date_manual_cbh_list.append(np.array(df['manual_cbh'][mask]))
    
    bp1 = ax1.boxplot(F_cre_sat_list, positions=F_cre_sat_postion_list, widths=0.25,
                        patch_artist=True, boxprops=dict(facecolor='tab:red', color='tab:red'),
                        medianprops=dict(color='black'), showfliers=False, showmeans=True,
                        meanline=True, meanprops=dict(color='black', linestyle='--'))
    bp2 = ax1.boxplot(F_cre_manual_list, positions=F_cre_manual_postion_list, widths=0.25,
                        patch_artist=True, boxprops=dict(facecolor='tab:green', color='tab:blue'),
                        medianprops=dict(color='black'), showfliers=False, showmeans=True,
                        meanline=True, meanprops=dict(color='black', linestyle='--'))
        
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        for j in bp1[element]:
            j.set(color='r',)# linewidth=1.5)
        for j in bp2[element]:
            j.set(color='g',)# linewidth=1.5)
        for patch in bp1['boxes']:
            patch.set(facecolor='lightsalmon')  # Set the color of the box
        for patch in bp2['boxes']:
            patch.set(facecolor='palegreen')  # Set the color of the box
    # ax1.set_ylim(-122, 122)
    ax1.axhline(0, color='black', linestyle='--', linewidth=0.5)
    
    ax1.legend([bp1["boxes"][0], bp2["boxes"][0]], ['satellite cloud', 'measured cloud'], loc='best', fontsize=legend_size)
    
    date_postion_list = np.array(date_postion_list)
    ax1.set_xticks(date_postion_list, date_label_list,)# rotation=45, ha='right')
    ax1.set_ylabel("LW CRE\n(W m$^{-2}$)", fontsize=label_size)
    ax1.set_xlabel("Date", fontsize=label_size)
    
    ax1.tick_params('both', labelsize=label_size-2)
    # fig.suptitle(f"Clouds on {date}", fontsize=label_size+2, y=0.98)
    fig.tight_layout()
    fig.savefig(f"fig/clouds/CRE_compare_LW_{fig_tag}.png", dpi=300, bbox_inches='tight')
    
    results = {'date_label_list': date_label_list,
               'F_cre_sat_list': F_cre_sat_list,
               'F_cre_manual_list': F_cre_manual_list,
               'F_cre_sat_postion_list': F_cre_sat_postion_list,
               'F_cre_manual_postion_list': F_cre_manual_postion_list,
               'date_postion_list': date_postion_list,
               'sat_cer_list': date_sat_cer_list,
               'sat_cwp_list': date_sat_cwp_list,
               'sat_cot_list': date_sat_cot_list,
                'sat_cth_list': date_sat_cth_list,
                'sat_cbh_list': date_sat_cbh_list,
               'manual_cer_list': date_manual_cer_list,
               'manual_cwp_list': date_manual_cwp_list,
               'manual_cot_list': date_manual_cot_list,
                'manual_cth_list': date_manual_cth_list,
                'manual_cbh_list': date_manual_cbh_list,
               }
    
    return results


def plot_combine_results(sw_result, lw_result, fig_tag='test'):
    """
    Plot the combined results of SW and LW CRE.
    """
    F_cre_sat_combined_list = []
    F_cre_manual_combined_list = []
    F_cre_sat_postion_list = []
    F_cre_manual_postion_list = []
    date_postion_list = []
    date_label_list = []
    
    fig = plt.figure(figsize=(5, 2.9))
    ax1 = fig.add_subplot(111)
    
    for i in range(len(sw_result['date_label_list'])):
        # print("i:", i)
        # print("sw_result['F_cre_sat_list'][i] shape:", np.shape(sw_result['F_cre_sat_list'][i]))
        # print("lw_result['F_cre_sat_list'][i] shape:", np.shape(lw_result['F_cre_sat_list'][i]))
        
        
        F_cre_sat_combined_list.append(sw_result['F_cre_sat_list'][i] + lw_result['F_cre_sat_list'][i])
        F_cre_manual_combined_list.append(sw_result['F_cre_manual_list'][i] + lw_result['F_cre_manual_list'][i])
        
        date_label_list.append(sw_result['date_label_list'][i])
        F_cre_sat_postion_list.append(sw_result['F_cre_sat_postion_list'][i])
        F_cre_manual_postion_list.append(sw_result['F_cre_manual_postion_list'][i])
        date_postion_list.append(sw_result['date_postion_list'][i])
    
    # Combined CRE
    bp1 = ax1.boxplot(F_cre_sat_combined_list, positions=F_cre_sat_postion_list, widths=0.25,
                        patch_artist=True, boxprops=dict(facecolor='tab:red', color='tab:red'),
                        medianprops=dict(color='black'), showfliers=False, showmeans=True,
                        meanline=True, meanprops=dict(color='black', linestyle='--'))
    bp2 = ax1.boxplot(F_cre_manual_combined_list, positions=F_cre_manual_postion_list, widths=0.25,
                        patch_artist=True, boxprops=dict(facecolor='tab:green', color='tab:blue'),
                        medianprops=dict(color='black'), showfliers=False, showmeans=True,
                        meanline=True, meanprops=dict(color='black', linestyle='--'))
    
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        for j in bp1[element]:
            j.set(color='r',)# linewidth=1.5)
        for j in bp2[element]:
            j.set(color='g',)# linewidth=1.5)
        for patch in bp1['boxes']:
            patch.set(facecolor='lightsalmon')  # Set the color of the box
        for patch in bp2['boxes']:
            patch.set(facecolor='palegreen')  # Set the color of the box
    # ax1.set_ylim(-122, 122)
    ax1.axhline(0, color='black', linestyle='--', linewidth=0.5)
    
    # ax1.legend([bp1["boxes"][0], bp2["boxes"][0]], ['satellite cloud', 'measured cloud'], loc='best', fontsize=legend_size)
    
    date_postion_list = np.array(date_postion_list)
    ax1.set_xticks(date_postion_list, date_label_list,)# rotation=45, ha='right')
    ax1.set_ylabel("Net CRE\n(W m$^{-2}$)", fontsize=label_size)
    ax1.set_xlabel("Date", fontsize=label_size)
    
    ax1.tick_params('both', labelsize=label_size-2)
    # fig.suptitle(f"Clouds on {date}", fontsize=label_size+2, y=0.98)
    fig.tight_layout()
    fig.savefig(f"fig/clouds/CRE_compare_NET_{fig_tag}.png", dpi=300, bbox_inches='tight')



def hypothetical_cre(fname_sw, fname_lw,
                          fig_tag='test',
                          overwrite=False
                            ):


    
    seg_list = []
    alt_list = []
    date_list = []
    sza_list = []
    # ssfr_zen_flux_list = []
    # ssfr_nad_flux_list = []
    # hsr1_total_flux_list = []
    # hsr1_diff_flux_list = []
    # bbr_up_flux_list = []
    # bbr_down_flux_list = []
    # bbr_sky_T_list = []
    # kt19_T_list = []
    sat_cot_list = []
    sat_cth_list = []
    sat_cwp_list = []
    sat_cer_list = []
    sat_f_down_list = []
    sat_f_down_dir_list = []
    sat_f_down_diff_list = []
    sat_f_up_list = []
    manual_cot_list = []
    manual_cth_list = []
    manual_cwp_list = []
    manual_cer_list = []
    manual_f_down_list = []
    manual_f_down_dir_list = []
    manual_f_down_diff_list = []
    manual_f_up_list = []
    clear_f_down_list = []
    clear_f_down_dir_list = []
    clear_f_down_diff_list = []
    clear_f_up_list = []
    
    dwvl_nm = 1
    
    sw_z_ind = 0
    with h5py.File(fname_sw, 'r') as f:
        f_zpt_sw = f['zpt'][...]
        f_alb_sw = f['alb'][...]
        f_sza_sw = f['sza'][...]
        f_down_1d_sw = f['f_down'][...][:, :, sw_z_ind]
        f_down_dir_1d_sw = f['f_down_dir'][...][:, :, sw_z_ind]
        f_down_diff_1d_sw = f['f_down_diff'][...][:, :, sw_z_ind]
        f_up_1d_sw = f['f_up'][...][:, :, sw_z_ind]
        f_cth_sw = f['cth'][...]
        f_cbh_sw = f['cbh'][...]
        f_cot_sw = f['cot'][...]
        f_cwp_sw = f['cwp'][...] * 1000  # convert to g m^-2
        f_cer_sw = f['cer'][...]
        
        f_down_sw = np.sum(f_down_1d_sw*dwvl_nm, axis=1)  # sum over wavelengths
        f_up_sw = np.sum(f_up_1d_sw*dwvl_nm, axis=1)
        
        print("f_down_1d_sw shape:", f_down_1d_sw.shape)
        print("f_down_sw shape:", f_down_sw.shape)
        
        clear_sky_mask = f_cot_sw == 0.0  # clear sky mask based on COT
        
        f_down_sw_clear = f_down_sw[clear_sky_mask]
        f_up_sw_clear = f_up_sw[clear_sky_mask]
        f_down_sw_cloud = f_down_sw#[~clear_sky_mask]
        f_up_sw_cloud = f_up_sw#[~clear_sky_mask]
        
    
    # lw_out = pd.read_csv('output_0000.txt', sep='\s+', header=None)
    # lw_wvl = lw_out.iloc[:, 0].values.astype(float)  # first col is wavelength
    # dlw_wvl_nm = lw_wvl[1:] - lw_wvl[:-1]  # wavelength step in nm
    # dlw_wvl_nm = np.concatenate((np.array([dlw_wvl_nm[0]]), dlw_wvl_nm))  # add a zero for the last wavelength
    
    with h5py.File(fname_lw, 'r') as f:
        f_zpt_lw = f['zpt'][...]
        f_alb_lw = f['alb'][...]
        f_sza_lw = f['sza'][...]
        f_down_1d_lw = f['f_down'][...][:, 0, 0]*1000
        f_down_dir_lw = f['f_down_dir'][...][:, 0, 0]*1000
        f_down_diff_lw = f['f_down_diff'][...][:, 0, 0]*1000
        f_up_1d_lw = f['f_up'][...][:, 0, 0]*1000
        # f_down_1d_lw = f['f_down'][...][:, :]*1000
        # f_down_dir_lw = f['f_down_dir'][...][:, :]*1000
        # f_down_diff_lw = f['f_down_diff'][...][:, :]*1000
        # f_up_1d_lw = f['f_up'][...][:, :]*1000
        f_cth_lw = f['cth'][...]
        f_cbh_lw = f['cbh'][...]
        f_cot_lw = f['cot'][...]
        f_cwp_lw = f['cwp'][...]  * 1000  # convert to g m^-2
        f_cer_lw = f['cer'][...]
    
        # print("f['f_down'][...] shape:", f['f_down'][...].shape)
        # print("f_down_1d_lw shape:", f_down_1d_lw.shape)
        # sys.exit()
        # f_down_lw = np.sum(f_down_1d_lw*dlw_wvl_nm, axis=1)  # sum over wavelengths
        # f_up_lw = np.sum(f_up_1d_lw*dlw_wvl_nm, axis=1)
        
        f_down_lw = f_down_1d_lw
        f_up_lw = f_up_1d_lw
        
        clear_sky_mask = f_cot_lw == 0.0  # clear sky mask based on COT
        f_down_lw_clear = f_down_lw[clear_sky_mask]
        f_up_lw_clear = f_up_lw[clear_sky_mask]
        f_down_lw_cloud = f_down_lw#[~clear_sky_mask]
        f_up_lw_cloud = f_up_lw#[~clear_sky_mask]
    

        
    F_clear_sw = f_down_sw_clear - f_up_sw_clear
    F_cloud_sw = f_down_sw_cloud - f_up_sw_cloud
    F_clear_lw = f_down_lw_clear - f_up_lw_clear
    F_cloud_lw = f_down_lw_cloud - f_up_lw_cloud
    
    
    swCRE = F_cloud_sw - F_clear_sw
    lwCRE = F_cloud_lw - F_clear_lw
    
    F_net = swCRE + lwCRE
    
    cwp_arr = np.array(f_cwp_lw).flatten()  # CWP for cloudy pixels
    swCRE = np.array(swCRE).flatten()
    lwCRE = np.array(lwCRE).flatten()
    F_net = np.array(F_net).flatten()

    cer_x = np.max(f_cer_lw) # CER for cloudy pixels
    cth_x = np.max(f_cth_lw) # CTH for cloudy pixels
    cbh_x = np.max(f_cbh_lw) # CBH for cloudy pixels
    cgt_x = cth_x-cbh_x
    
    fig = plt.figure(figsize=(7, 5))
    ax1 = fig.add_subplot(111)
    ax1.plot(cwp_arr, swCRE, label='SW', color='tab:red')
    ax1.plot(cwp_arr, lwCRE, label='LW', color='tab:blue')
    ax1.plot(cwp_arr, F_net, label='Net', color='tab:green')
    ax1.set_xlabel('CWP (g m$^{-2}$)', fontsize=label_size)
    ax1.set_ylabel('CRE (W m$^{-2}$)', fontsize=label_size)
    ax1.set_title(f'Hypothetical Cloud Radiative Effect\nCER {cer_x:.1f} $\mu$m, CTH {cth_x:.1f} km, CGT {cgt_x*1000:.0f} m ',
                  fontsize=label_size+2)
    ax1.tick_params('both', labelsize=label_size-2)
    ax1.legend(loc='best', fontsize=legend_size)
    ax1.axhline(0, color='grey', linestyle='--', linewidth=0.5)
    fig.tight_layout()
    fig.savefig(f"fig/clouds/hypothetical_CRE_{fig_tag}.png", dpi=300, bbox_inches='tight')
    sys.exit()
    
    # print("F_cre_sat_cloud:", F_cre_sat_cloud)
    # print("F_cre_manual_cloud:", F_cre_manual_cloud)
    
    # sys.exit()
    
    F_cre_sat_list = []
    F_cre_manual_list = []
    F_cre_sat_postion_list = []
    F_cre_manual_postion_list = []
    date_postion_list = []
    date_label_list = []
    
    # plot the results
    fig = plt.figure(figsize=(5, 2.9))
    ax1 = fig.add_subplot(111)
    
    color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
    for ind, j in enumerate(sorted(set(date_list))):
        mask = np.array(df['date']) == j
        print(f"Processing date: {j}, mask length: {np.sum(mask)}")

        mm = np.int(j[4:6])
        dd = np.int(j[6:8])
        date_label = f"{mm:d}/{dd:d}"
        
        # print()
        
        F_cre_sat_list.append(F_cre_sat_cloud[mask])
        F_cre_manual_list.append(F_cre_manual_cloud[mask])
        
        F_cre_sat_postion_list.append(ind*2-0.25)
        F_cre_manual_postion_list.append(ind*2+0.25)
        date_postion_list.append(ind*2)
        date_label_list.append(date_label)
    
    bp1 = ax1.boxplot(F_cre_sat_list, positions=F_cre_sat_postion_list, widths=0.25,
                        patch_artist=True, boxprops=dict(facecolor='tab:red', color='tab:red'),
                        medianprops=dict(color='black'), showfliers=False, showmeans=True,
                        meanline=True, meanprops=dict(color='black', linestyle='--'))
    bp2 = ax1.boxplot(F_cre_manual_list, positions=F_cre_manual_postion_list, widths=0.25,
                        patch_artist=True, boxprops=dict(facecolor='tab:green', color='tab:blue'),
                        medianprops=dict(color='black'), showfliers=False, showmeans=True,
                        meanline=True, meanprops=dict(color='black', linestyle='--'))
        
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        for j in bp1[element]:
            j.set(color='r',)# linewidth=1.5)
        for j in bp2[element]:
            j.set(color='g',)# linewidth=1.5)
        for patch in bp1['boxes']:
            patch.set(facecolor='lightsalmon')  # Set the color of the box
        for patch in bp2['boxes']:
            patch.set(facecolor='palegreen')  # Set the color of the box
    # ax1.set_ylim(-122, 122)
    ax1.axhline(0, color='black', linestyle='--', linewidth=0.5)
    
    # ax1.legend([bp1["boxes"][0], bp2["boxes"][0]], ['satellite cloud', 'measured cloud'], loc='best', fontsize=legend_size)
    
    date_postion_list = np.array(date_postion_list)
    ax1.set_xticks(date_postion_list, date_label_list,)# rotation=45, ha='right')
    ax1.set_ylabel("SW CRE\n(W m$^{-2}$)", fontsize=label_size)
    ax1.set_xlabel("Date", fontsize=label_size)
    
    ax1.tick_params('both', labelsize=label_size-2)
    # fig.suptitle(f"Clouds on {date}", fontsize=label_size+2, y=0.98)
    fig.tight_layout()
    fig.savefig(f"fig/clouds/CRE_compare_SW_{fig_tag}.png", dpi=300, bbox_inches='tight')
    
    
    results = {'date_label_list': date_label_list,
               'F_cre_sat_list': F_cre_sat_list,
               'F_cre_manual_list': F_cre_manual_list,
               'F_cre_sat_postion_list': F_cre_sat_postion_list,
               'F_cre_manual_postion_list': F_cre_manual_postion_list,
               'date_postion_list': date_postion_list}
    
    return results


def hypothetical_cre_with_obs(fname_sw, fname_lw,
                              sw_result, lw_result,
                          fig_tag='test',
                          overwrite=False
                            ):


    
    seg_list = []
    alt_list = []
    date_list = []
    sza_list = []
    # ssfr_zen_flux_list = []
    # ssfr_nad_flux_list = []
    # hsr1_total_flux_list = []
    # hsr1_diff_flux_list = []
    # bbr_up_flux_list = []
    # bbr_down_flux_list = []
    # bbr_sky_T_list = []
    # kt19_T_list = []
    sat_cot_list = []
    sat_cth_list = []
    sat_cwp_list = []
    sat_cer_list = []
    sat_f_down_list = []
    sat_f_down_dir_list = []
    sat_f_down_diff_list = []
    sat_f_up_list = []
    manual_cot_list = []
    manual_cth_list = []
    manual_cwp_list = []
    manual_cer_list = []
    manual_f_down_list = []
    manual_f_down_dir_list = []
    manual_f_down_diff_list = []
    manual_f_up_list = []
    clear_f_down_list = []
    clear_f_down_dir_list = []
    clear_f_down_diff_list = []
    clear_f_up_list = []
    
    dwvl_nm = 1
    
    sw_z_ind = 0
    with h5py.File(fname_sw, 'r') as f:
        f_zpt_sw = f['zpt'][...]
        f_alb_sw = f['alb'][...]
        f_sza_sw = f['sza'][...]
        f_down_1d_sw = f['f_down'][...][:, :, sw_z_ind]
        f_down_dir_1d_sw = f['f_down_dir'][...][:, :, sw_z_ind]
        f_down_diff_1d_sw = f['f_down_diff'][...][:, :, sw_z_ind]
        f_up_1d_sw = f['f_up'][...][:, :, sw_z_ind]
        f_cth_sw = f['cth'][...]
        f_cbh_sw = f['cbh'][...]
        f_cot_sw = f['cot'][...]
        f_cwp_sw = f['cwp'][...] * 1000  # convert to g m^-2
        f_cer_sw = f['cer'][...]
        
        f_down_sw = np.sum(f_down_1d_sw*dwvl_nm, axis=1)  # sum over wavelengths
        f_up_sw = np.sum(f_up_1d_sw*dwvl_nm, axis=1)
        
        print("f_down_1d_sw shape:", f_down_1d_sw.shape)
        print("f_down_sw shape:", f_down_sw.shape)
        
        clear_sky_mask = f_cot_sw == 0.0  # clear sky mask based on COT
        
        f_down_sw_clear = f_down_sw[clear_sky_mask]
        f_up_sw_clear = f_up_sw[clear_sky_mask]
        f_down_sw_cloud = f_down_sw#[~clear_sky_mask]
        f_up_sw_cloud = f_up_sw#[~clear_sky_mask]
        
    
    # lw_out = pd.read_csv('output_0000.txt', sep='\s+', header=None)
    # lw_wvl = lw_out.iloc[:, 0].values.astype(float)  # first col is wavelength
    # dlw_wvl_nm = lw_wvl[1:] - lw_wvl[:-1]  # wavelength step in nm
    # dlw_wvl_nm = np.concatenate((np.array([dlw_wvl_nm[0]]), dlw_wvl_nm))  # add a zero for the last wavelength
    
    with h5py.File(fname_lw, 'r') as f:
        f_zpt_lw = f['zpt'][...]
        f_alb_lw = f['alb'][...]
        f_sza_lw = f['sza'][...]
        f_down_1d_lw = f['f_down'][...][:, 0, 0]*1000
        f_down_dir_lw = f['f_down_dir'][...][:, 0, 0]*1000
        f_down_diff_lw = f['f_down_diff'][...][:, 0, 0]*1000
        f_up_1d_lw = f['f_up'][...][:, 0, 0]*1000
        # f_down_1d_lw = f['f_down'][...][:, :]*1000
        # f_down_dir_lw = f['f_down_dir'][...][:, :]*1000
        # f_down_diff_lw = f['f_down_diff'][...][:, :]*1000
        # f_up_1d_lw = f['f_up'][...][:, :]*1000
        f_cth_lw = f['cth'][...]
        f_cbh_lw = f['cbh'][...]
        f_cot_lw = f['cot'][...]
        f_cwp_lw = f['cwp'][...]  * 1000  # convert to g m^-2
        f_cer_lw = f['cer'][...]
    
        # print("f['f_down'][...] shape:", f['f_down'][...].shape)
        # print("f_down_1d_lw shape:", f_down_1d_lw.shape)
        # sys.exit()
        # f_down_lw = np.sum(f_down_1d_lw*dlw_wvl_nm, axis=1)  # sum over wavelengths
        # f_up_lw = np.sum(f_up_1d_lw*dlw_wvl_nm, axis=1)
        
        f_down_lw = f_down_1d_lw
        f_up_lw = f_up_1d_lw
        
        clear_sky_mask = f_cot_lw == 0.0  # clear sky mask based on COT
        f_down_lw_clear = f_down_lw[clear_sky_mask]
        f_up_lw_clear = f_up_lw[clear_sky_mask]
        f_down_lw_cloud = f_down_lw#[~clear_sky_mask]
        f_up_lw_cloud = f_up_lw#[~clear_sky_mask]
    

        
    F_clear_sw = f_down_sw_clear - f_up_sw_clear
    F_cloud_sw = f_down_sw_cloud - f_up_sw_cloud
    F_clear_lw = f_down_lw_clear - f_up_lw_clear
    F_cloud_lw = f_down_lw_cloud - f_up_lw_cloud
    
    
    swCRE = F_cloud_sw - F_clear_sw
    lwCRE = F_cloud_lw - F_clear_lw
    
    F_net = swCRE + lwCRE
    
    cwp_arr = np.array(f_cwp_lw).flatten()  # CWP for cloudy pixels
    swCRE = np.array(swCRE).flatten()
    lwCRE = np.array(lwCRE).flatten()
    F_net = np.array(F_net).flatten()

    cer_x = np.max(f_cer_lw) # CER for cloudy pixels
    cth_x = np.max(f_cth_lw) # CTH for cloudy pixels
    cbh_x = np.max(f_cbh_lw) # CBH for cloudy pixels
    cgt_x = cth_x-cbh_x
    
    
    
    date_label_list = sw_result['date_label_list']
    
    sat_cer_list = sw_result['sat_cer_list']
    sat_cwp_list = sw_result['sat_cwp_list']
    sat_cot_list = sw_result['sat_cot_list']
    sat_cth_list = sw_result['sat_cth_list']
    sat_cbh_list = sw_result['sat_cbh_list']
    manual_cer_list = lw_result['manual_cer_list']
    manual_cwp_list = lw_result['manual_cwp_list']
    manual_cot_list = lw_result['manual_cot_list']
    sat_cth_list = sw_result['sat_cth_list']
    manual_cth_list = lw_result['manual_cth_list']
    sat_cbh_list = sw_result['sat_cbh_list']
    manual_cbh_list = lw_result['manual_cbh_list']
    

    obs_net_cre_sat_all_list = []
    obs_net_cre_manual_all_list = []
    obs_net_cre_sat_mean_list = []
    obs_net_cre_manual_mean_list = []
    obs_net_cre_sat_std_list = []
    obs_net_cre_manual_std_list = []
    sat_lwp_list = []
    manual_lwp_list = []
    sat_cer_mean_list = []
    manual_cer_mean_list = []
    sat_cth_mean_list = []
    manual_cth_mean_list = []
    sat_cbh_mean_list = []
    manual_cbh_mean_list = []
    
    for i in range(len(sat_cwp_list)):
        
        obs_net_cre_sat_all = sw_result['F_cre_sat_list'][i] + lw_result['F_cre_sat_list'][i]
        obs_net_cre_manual_all = sw_result['F_cre_manual_list'][i] + lw_result['F_cre_manual_list'][i]
        obs_net_cre_sat_mean = np.nanmean(obs_net_cre_sat_all)
        obs_net_cre_manual_mean = np.nanmean(obs_net_cre_manual_all)
        obs_net_cre_sat_std = np.nanstd(obs_net_cre_sat_all)
        obs_net_cre_manual_std = np.nanstd(obs_net_cre_manual_all)
        sat_cwp_nanmean = np.nanmean(sat_cwp_list[i])
        manual_cwp_nanmean = np.nanmean(manual_cwp_list[i])
        sat_cer_nanmean = np.nanmean(sat_cer_list[i])
        manual_cer_nanmean = np.nanmean(manual_cer_list[i])
        
        
        obs_net_cre_sat_all_list.append(obs_net_cre_sat_all)
        obs_net_cre_manual_all_list.append(obs_net_cre_manual_all)
        obs_net_cre_sat_mean_list.append(obs_net_cre_sat_mean)
        obs_net_cre_manual_mean_list.append(obs_net_cre_manual_mean)
        obs_net_cre_sat_std_list.append(obs_net_cre_sat_std)
        obs_net_cre_manual_std_list.append(obs_net_cre_manual_std)
        sat_lwp_list.append(np.nanmean(sat_cwp_list[i]))
        manual_lwp_list.append(np.nanmean(manual_cwp_list[i]))
        sat_cer_mean_list.append(sat_cer_nanmean)
        manual_cer_mean_list.append(manual_cer_nanmean)
        sat_cth_mean_list.append(np.nanmean(sat_cth_list[i]))
        manual_cth_mean_list.append(np.nanmean(manual_cth_list[i]))
        sat_cbh_mean_list.append(np.nanmean(sat_cth_list[i]))
        manual_cbh_mean_list.append(np.nanmean(manual_cth_list[i]))
    
    
    print("date_label_list:", date_label_list)
    print("sat_cer_mean_list:", sat_cer_mean_list)
    print("manual_cer_mean_list:", manual_cer_mean_list)
    print("sat_cth_mean_list:", sat_cth_mean_list)
    print("manual_cth_mean_list:", manual_cth_mean_list)
    
    fig = plt.figure(figsize=(6, 5.5))
    ax1 = fig.add_subplot(111)
    l1 = ax1.plot(cwp_arr, swCRE, label='SW', color='tab:blue')
    l2 = ax1.plot(cwp_arr, lwCRE, label='LW', color='tab:red')
    l3 = ax1.plot(cwp_arr, F_net, label='Net', color='tab:green')
    ax1.set_xlabel('CWP (g m$^{-2}$)', fontsize=label_size)
    ax1.set_ylabel('Net CRE (W m$^{-2}$)', fontsize=label_size)
    
    ax1.text(30, 90, f"CER {cer_x:.1f} $\mu$m, CTH {cth_x:.1f} km, CGT {cgt_x*1000:.0f} m ", fontsize=label_size-2,)
    
    
    erl1 = ax1.errorbar(sat_lwp_list, obs_net_cre_sat_mean_list,
                    yerr=obs_net_cre_sat_std_list, fmt='o', color='tab:orange', label='Satellite cloud CRE', capsize=8, elinewidth=1,)# markeredgewidth=2)
    erl2 = ax1.errorbar(manual_lwp_list, obs_net_cre_manual_mean_list,
                    yerr=obs_net_cre_manual_std_list, fmt='s', color='tab:purple', label='Measured cloud CRE', capsize=8, elinewidth=1,)# markeredgewidth=2)
    
    ax1.tick_params('both', labelsize=label_size-2)
    # ax1.legend(fontsize=legend_size, ncol=2,
    #             bbox_to_anchor=(0.5, -0.2), loc='upper center'
                
    #            )# frameon=False)
    ax1.legend([l1[0], erl1, l2[0], erl2, l3[0]],
               ['SW CRE', 'Satellite cloud CRE', 'LW CRE', 'Measured cloud CRE', 'Net CRE',],
               ncol=3, bbox_to_anchor=(0.5, -0.17), loc='upper center')
    ax1.axhline(0, color='grey', linestyle='--', linewidth=0.5)
    
    ax1.set_xlim(0, 200)
    fig.suptitle(f'Hypothetical Cloud Radiative Effect',
                  fontsize=label_size+2)
    
    fig.tight_layout()
    fig.savefig(f"fig/clouds/hypothetical_CRE_with_obs{fig_tag}.png", dpi=300, bbox_inches='tight')
    
    
    fig = plt.figure(figsize=(7, 5.5))
    ax1 = fig.add_subplot(111)
    l1 = ax1.plot(cwp_arr, swCRE, label='SW', color='tab:red')
    l2 = ax1.plot(cwp_arr, lwCRE, label='LW', color='tab:blue')
    l3 = ax1.plot(cwp_arr, F_net, label='Net', color='tab:green')
    ax1.set_xlabel('CWP (g m$^{-2}$)', fontsize=label_size)
    ax1.set_ylabel('Net CRE (W m$^{-2}$)', fontsize=label_size)
    
    ax1.text(30, 100, f"CER {cer_x:.1f} $\mu$m, CTH {cth_x:.1f} km, CGT {cgt_x*1000:.0f} m ", fontsize=label_size-2,)
    
    
    erl1 = ax1.errorbar(sat_lwp_list, obs_net_cre_sat_mean_list,
                    yerr=obs_net_cre_sat_std_list, fmt='o', c='tab:orange', label='Satellite cloud CRE', capsize=8, elinewidth=1, ms=5, mfc=None, zorder=2)# markeredgewidth=2)
    erl2 = ax1.errorbar(manual_lwp_list, obs_net_cre_manual_mean_list,
                    yerr=obs_net_cre_manual_std_list, fmt='s', c='tab:purple', label='Measured cloud CRE', capsize=8, elinewidth=1, ms=5, mfc=None, zorder=2)# markeredgewidth=2)

    sc1 = ax1.scatter(sat_lwp_list, obs_net_cre_sat_mean_list, c=sat_cer_mean_list, marker='o', label='Satellite CER', s=70, cmap='jet', vmin=5, vmax=15, zorder=10)
    sc2 = ax1.scatter(manual_lwp_list, obs_net_cre_manual_mean_list, c=manual_cer_mean_list, marker='s', label='Measured CER', s=70, cmap='jet', vmin=5, vmax=15, zorder=10)
    
    ax1.tick_params('both', labelsize=label_size-2)
    # ax1.legend(fontsize=legend_size, ncol=2,
    #             bbox_to_anchor=(0.5, -0.2), loc='upper center'
    
    cbar = fig.colorbar(sc1, ax=ax1, orientation='vertical', pad=0.05, extend='both', ticks=np.arange(5, 16, 3))
    cbar.set_label('CER ($\mu$m)', fontsize=label_size)
    cbar.ax.tick_params(labelsize=label_size-2)
    #            )# frameon=False)
    ax1.legend([l1[0], erl1, l2[0], erl2, l3[0]],
               ['SW CRE', 'Satellite cloud CRE', 'LW CRE', 'Measured cloud CRE', 'Net CRE',],
               ncol=3, bbox_to_anchor=(0.5, -0.17), loc='upper center')
    ax1.axhline(0, color='grey', linestyle='--', linewidth=0.5)
    
    ax1.set_xlim(0, 200)
    fig.suptitle(f'Hypothetical Cloud Radiative Effect',
                  fontsize=label_size+2)
    
    fig.tight_layout()
    fig.savefig(f"fig/clouds/hypothetical_CRE_with_obs{fig_tag}_with_CER.png", dpi=300, bbox_inches='tight')
    
    
    
    fig = plt.figure(figsize=(7, 5.5))
    ax1 = fig.add_subplot(111)
    l1 = ax1.plot(cwp_arr, swCRE, label='SW', color='tab:red')
    l2 = ax1.plot(cwp_arr, lwCRE, label='LW', color='tab:blue')
    l3 = ax1.plot(cwp_arr, F_net, label='Net', color='tab:green')
    ax1.set_xlabel('CWP (g m$^{-2}$)', fontsize=label_size)
    ax1.set_ylabel('Net CRE (W m$^{-2}$)', fontsize=label_size)
    
    ax1.text(30, 100, f"CER {cer_x:.1f} $\mu$m, CTH {cth_x:.1f} km, CGT {cgt_x*1000:.0f} m ", fontsize=label_size-2,)
    
    
    erl1 = ax1.errorbar(sat_lwp_list, obs_net_cre_sat_mean_list,
                    yerr=obs_net_cre_sat_std_list, fmt='o', c='tab:orange', label='Satellite cloud CRE', capsize=8, elinewidth=1, ms=5, mfc=None, zorder=2)# markeredgewidth=2)
    erl2 = ax1.errorbar(manual_lwp_list, obs_net_cre_manual_mean_list,
                    yerr=obs_net_cre_manual_std_list, fmt='s', c='tab:purple', label='Measured cloud CRE', capsize=8, elinewidth=1, ms=5, mfc=None, zorder=2)# markeredgewidth=2)

    sc1 = ax1.scatter(sat_lwp_list, obs_net_cre_sat_mean_list, c=sat_cth_mean_list, marker='o', label='Satellite CER', s=70, cmap='jet', vmin=0, vmax=4, zorder=10)
    sc2 = ax1.scatter(manual_lwp_list, obs_net_cre_manual_mean_list, c=manual_cth_mean_list, marker='s', label='Measured CER', s=70, cmap='jet', vmin=0, vmax=4, zorder=10)
    
    ax1.tick_params('both', labelsize=label_size-2)
    # ax1.legend(fontsize=legend_size, ncol=2,
    #             bbox_to_anchor=(0.5, -0.2), loc='upper center'
    
    cbar = fig.colorbar(sc1, ax=ax1, orientation='vertical', pad=0.05, extend='both',)# ticks=np.arange(5, 16, 3))
    cbar.set_label('CTH (km)', fontsize=label_size)
    cbar.ax.tick_params(labelsize=label_size-2)
    #            )# frameon=False)
    ax1.legend([l1[0], erl1, l2[0], erl2, l3[0]],
               ['SW CRE', 'Satellite cloud CRE', 'LW CRE', 'Measured cloud CRE', 'Net CRE',],
               ncol=3, bbox_to_anchor=(0.5, -0.17), loc='upper center')
    ax1.axhline(0, color='grey', linestyle='--', linewidth=0.5)
    
    ax1.set_xlim(0, 200)
    fig.suptitle(f'Hypothetical Cloud Radiative Effect',
                  fontsize=label_size+2)
    
    fig.tight_layout()
    fig.savefig(f"fig/clouds/hypothetical_CRE_with_obs{fig_tag}_with_CTH.png", dpi=300, bbox_inches='tight')
    
    fig = plt.figure(figsize=(7, 5.5))
    ax1 = fig.add_subplot(111)
    l1 = ax1.plot(cwp_arr, swCRE, label='SW', color='tab:red')
    l2 = ax1.plot(cwp_arr, lwCRE, label='LW', color='tab:blue')
    l3 = ax1.plot(cwp_arr, F_net, label='Net', color='tab:green')
    ax1.set_xlabel('CWP (g m$^{-2}$)', fontsize=label_size)
    ax1.set_ylabel('Net CRE (W m$^{-2}$)', fontsize=label_size)
    
    ax1.text(30, 100, f"CER {cer_x:.1f} $\mu$m, CTH {cth_x:.1f} km, CGT {cgt_x*1000:.0f} m ", fontsize=label_size-2,)
    
    
    erl1 = ax1.errorbar(sat_lwp_list, obs_net_cre_sat_mean_list,
                    yerr=obs_net_cre_sat_std_list, fmt='o', c='tab:orange', label='Satellite cloud CRE', capsize=8, elinewidth=1, ms=5, mfc=None, zorder=2)# markeredgewidth=2)
    erl2 = ax1.errorbar(manual_lwp_list, obs_net_cre_manual_mean_list,
                    yerr=obs_net_cre_manual_std_list, fmt='s', c='tab:purple', label='Measured cloud CRE', capsize=8, elinewidth=1, ms=5, mfc=None, zorder=2)# markeredgewidth=2)

    sc1 = ax1.scatter(sat_lwp_list, obs_net_cre_sat_mean_list, c=sat_cbh_mean_list, marker='o', label='Satellite CBH', s=70, cmap='jet', vmin=0, vmax=3, zorder=10)
    sc2 = ax1.scatter(manual_lwp_list, obs_net_cre_manual_mean_list, c=manual_cbh_mean_list, marker='s', label='Measured CBH', s=70, cmap='jet', vmin=0, vmax=3, zorder=10)
    
    ax1.tick_params('both', labelsize=label_size-2)
    # ax1.legend(fontsize=legend_size, ncol=2,
    #             bbox_to_anchor=(0.5, -0.2), loc='upper center'
    
    cbar = fig.colorbar(sc1, ax=ax1, orientation='vertical', pad=0.05, extend='both',)# ticks=np.arange(5, 16, 3))
    cbar.set_label('CBH (km)', fontsize=label_size)
    cbar.ax.tick_params(labelsize=label_size-2)
    #            )# frameon=False)
    ax1.legend([l1[0], erl1, l2[0], erl2, l3[0]],
               ['SW CRE', 'Satellite cloud CRE', 'LW CRE', 'Measured cloud CRE', 'Net CRE',],
               ncol=3, bbox_to_anchor=(0.5, -0.17), loc='upper center')
    ax1.axhline(0, color='grey', linestyle='--', linewidth=0.5)
    
    ax1.set_xlim(0, 200)
    fig.suptitle(f'Hypothetical Cloud Radiative Effect',
                  fontsize=label_size+2)
    
    fig.tight_layout()
    fig.savefig(f"fig/clouds/hypothetical_CRE_with_obs{fig_tag}_with_CBH.png", dpi=300, bbox_inches='tight')
    
    return None




if __name__ == '__main__':

   
    
    #"""
    sw_result = flt_trk_cre_2lay_sw(dates=['20240603', '20240607', '20240606', '20240613'],
                          case_tag=['cloudy_track_1_cre', 'cloudy_track_2_cre', 'cloudy_track_2_cre', 'cloudy_track_1_cre'],
                          separate_heights=[600, 300, 300, 300],
                          fig_tag='test_sfc_cre',
                          overwrite=False
                            )
    
    lw_result = flt_trk_cre_2lay_lw(dates=['20240603', '20240607', '20240606', '20240613'],
                        #   case_tag=['cloudy_track_1_cre', 'cloudy_track_2_cre', 'cloudy_track_2_cre', 'cloudy_track_1_cre'],
                          case_tag=['cloudy_track_1_cre', 'cloudy_track_2_cre', 'cloudy_track_2_cre', 'cloudy_track_1_cre'],
                          separate_heights=[600, 300, 300, 300],
                          fig_tag='test_sfc_cre',
                          overwrite=False
                            )
    
    
    plot_combine_results(sw_result, lw_result, fig_tag='test_sfc_cre')
    #"""
    
    # hypothetical_cre('flt_trk_lrt_para-lrt-20240531-test_low_cld_high_alb-hypothetical.h5', 
    #                  'flt_trk_lrt_para-lrt-20240531-test_low_cld_high_alb-hypothetical-lw.h5',
    #                       fig_tag='test',
    #                       overwrite=False
    #                         )

    # hypothetical_cre('flt_trk_lrt_para-lrt-20240531-test_low_cld_high_alb-hypothetical-sw.h5', 
    #                  'flt_trk_lrt_para-lrt-20240531-test_low_cld_high_alb-hypothetical-lw.h5',
    #                       fig_tag='test',
    #                       overwrite=False
    #                         )
    
    hypothetical_cre_with_obs('flt_trk_lrt_para-lrt-20240531-test_low_cld_high_alb-hypothetical-sw.h5', 
                     'flt_trk_lrt_para-lrt-20240531-test_low_cld_high_alb-hypothetical-lw.h5',
                     sw_result, lw_result,
                          fig_tag='test',
                          overwrite=False
                            )
    
    # hypothetical_cre_with_obs('flt_trk_lrt_para-lrt-20240531-test_low_cld_high_alb_10um_2-hypothetical-sw.h5', 
    #                  'flt_trk_lrt_para-lrt-20240531-test_low_cld_high_alb_10um-hypothetical-lw.h5',
    #                  sw_result, lw_result,
    #                       fig_tag='test_10um',
    #                       overwrite=False
    #                         )
    
    # hypothetical_cre('flt_trk_lrt_para-lrt-20240531-hypothetical_test_mid_cld_high_alb-hypothetical-sw.h5', 
    #                  'flt_trk_lrt_para-lrt-20240531-hypothetical_test_mid_cld_high_alb-hypothetical-lw.h5',
    #                       fig_tag='test',
    #                       overwrite=False
    #                         )