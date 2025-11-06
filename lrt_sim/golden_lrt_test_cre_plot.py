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
    fig = plt.figure(figsize=(5, 3.1))
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
            cot = f['cot'][...]
            cwp = f['cwp'][...]*1000  # convert to g m^-2
            cer = f['cer'][...]
            f_down = f['f_down'][...][:, :, 1] # select at TOA
            f_down_dir = f['f_down_dir'][...][:, :, 1]
            f_down_diff = f['f_down_diff'][...][:, :, 1]
            f_up = f['f_up'][...][:, :, 1]
            
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
        sat_cwp_list.extend(cwp)
        sat_cer_list.extend(cer)
        
        # process manual cloud data
        lrt_fname_full = lrt_fname.format(date, tag, manual_cloud_tag)
        if not os.path.exists(lrt_fname_full) or overwrite:
            raise FileExistsError(f"File {lrt_fname_full} does not exist...")
        
        with h5py.File(lrt_fname_full, 'r') as f:
            cth = f['cth'][...]  # convert km
            cot = f['cot'][...]
            cwp = f['cwp'][...]*1000  # convert to g m^-2
            cer = f['cer'][...]
            f_down = f['f_down'][...][:, :, 1]
            f_down_dir = f['f_down_dir'][...][:, :, 1]
            f_down_diff = f['f_down_diff'][...][:, :, 1]
            f_up = f['f_up'][...][:, :, 1]
            
        manual_f_down_list.extend(np.sum(f_down*dwvl_nm, axis=1))  # sum over wavelengths
        manual_f_down_dir_list.extend(np.sum(f_down_dir*dwvl_nm, axis=1))
        manual_f_down_diff_list.extend(np.sum(f_down_diff*dwvl_nm, axis=1))
        manual_f_up_list.extend(np.sum(f_up*dwvl_nm, axis=1))   
        
        cth = np.where(cth < 0, np.nan, cth)  # set negative values to NaN
        manual_cot_list.extend(cot)
        manual_cth_list.extend(cth)
        manual_cwp_list.extend(cwp)
        manual_cer_list.extend(cer)
        
        # process clear sky data
        lrt_fname_full = lrt_fname.format(date, tag, clear_tag)
        if not os.path.exists(lrt_fname_full) or overwrite:
            raise FileExistsError(f"File {lrt_fname_full} does not exist...")
        with h5py.File(lrt_fname_full, 'r') as f:
            f_down = f['f_down'][...][:, :, 1]
            f_down_dir = f['f_down_dir'][...][:, :, 1]
            f_down_diff = f['f_down_diff'][...][:, :, 1]
            f_up = f['f_up'][...][:, :, 1]
        
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
        'sat_cwp': sat_cwp_list,
        'sat_cer': sat_cer_list,
        'sat_f_down': sat_f_down_list,
        'sat_f_down_dir': sat_f_down_dir_list,
        'sat_f_down_diff': sat_f_down_diff_list,
        'sat_f_up': sat_f_up_list,
        'manual_cot': manual_cot_list,
        'manual_cth': manual_cth_list,
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
    
    # print("isinstance(sat_f_down_list, list):", isinstance(sat_f_down_list, list))
    
    # for cloud_list in [sat_cot_list, sat_cth_list, sat_cwp_list, sat_cer_list,
    #                    manual_cot_list, manual_cth_list, manual_cwp_list, manual_cer_list]:
    #     cloud_list = np.array(cloud_list)
    #     cloud_list[cloud_list<0] = np.nan  # set negative values to NaN
    
    # for ind, cloud_list in enumerate([date_list, alt_list, sza_list, seg_list,
    #                                 sat_cot_list, sat_cth_list, sat_cwp_list, sat_cer_list,
    #                                 manual_cot_list, manual_cth_list, manual_cwp_list, manual_cer_list,
    #                                 sat_f_down_list, sat_f_down_dir_list, sat_f_down_diff_list, sat_f_up_list,
    #                                 manual_f_down_list, manual_f_down_dir_list, manual_f_down_diff_list, manual_f_up_list,
    #                                 clear_f_down_list, clear_f_down_dir_list, clear_f_down_diff_list, clear_f_up_list]):
    #     print(f"Processing cloud list {ind} with length: {len(cloud_list)}")
    #     if isinstance(cloud_list, list):
    #         cloud_list = np.array(cloud_list)
    #         print("isinstance(sat_f_down_list, list):", isinstance(sat_f_down_list, list))
    #         print("isinstance(cloud_list, list):", isinstance(cloud_list, list))
    #     cloud_list = cloud_list[~sat_cot_nan_mask]
        
    # print("isinstance(sat_f_down_list, list):", isinstance(sat_f_down_list, list))
    
    # calculate the CRE at TOA
    # F_sat_cloud = sat_f_down_list - sat_f_up_list
    # F_manual_cloud = manual_f_down_list - manual_f_up_list
    # F_clear = clear_f_down_list - clear_f_up_list
    
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
    
    # plot the results
    fig = plt.figure(figsize=(5, 3.1))
    ax1 = fig.add_subplot(111)
    
    color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
    for ind, j in enumerate(sorted(set(date_list))):
        mask = np.array(df['date']) == j
        print(f"Processing date: {j}, mask length: {np.sum(mask)}")

        mm = np.int(j[4:6])
        dd = np.int(j[6:8])
        date_label = f"{mm:d}/{dd:d}"
        
        print()
        
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
    ax1.set_ylim(-122, 22)
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
            cot = f['cot'][...]
            cwp = f['cwp'][...]*1000  # convert to g m^-2
            cer = f['cer'][...]
            f_down = f['f_down'][...][:, :, 1]*1000 # select at TOA
            f_down_dir = f['f_down_dir'][...][:, :, 1]*1000
            f_down_diff = f['f_down_diff'][...][:, :, 1]*1000
            f_up = f['f_up'][...][:, :, 1]*1000
        
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
        sat_cwp_list.extend(cwp)
        sat_cer_list.extend(cer)
        
        # process manual cloud data
        lrt_fname_full = lrt_fname.format(date, tag, manual_cloud_tag)
        if not os.path.exists(lrt_fname_full) or overwrite:
            raise FileExistsError(f"File {lrt_fname_full} does not exist...")
        
        with h5py.File(lrt_fname_full, 'r') as f:
            cth = f['cth'][...]  # convert km
            cot = f['cot'][...]
            cwp = f['cwp'][...]*1000  # convert to g m^-2
            cer = f['cer'][...]
            f_down = f['f_down'][...][:, :, 1]*1000
            f_down_dir = f['f_down_dir'][...][:, :, 1]*1000
            f_down_diff = f['f_down_diff'][...][:, :, 1]*1000
            f_up = f['f_up'][...][:, :, 1]*1000
            
        manual_f_down_list.extend(np.sum(f_down*dwvl_nm, axis=1))  # sum over wavelengths
        manual_f_down_dir_list.extend(np.sum(f_down_dir*dwvl_nm, axis=1))
        manual_f_down_diff_list.extend(np.sum(f_down_diff*dwvl_nm, axis=1))
        manual_f_up_list.extend(np.sum(f_up*dwvl_nm, axis=1))   
        
        cth = np.where(cth < 0, np.nan, cth)  # set negative values to NaN
        manual_cot_list.extend(cot)
        manual_cth_list.extend(cth)
        manual_cwp_list.extend(cwp)
        manual_cer_list.extend(cer)
        
        # process clear sky data
        lrt_fname_full = lrt_fname.format(date, tag, clear_tag)
        if not os.path.exists(lrt_fname_full) or overwrite:
            raise FileExistsError(f"File {lrt_fname_full} does not exist...")
        with h5py.File(lrt_fname_full, 'r') as f:
            f_down = f['f_down'][...][:, :, 1]*1000
            f_down_dir = f['f_down_dir'][...][:, :, 1]*1000
            f_down_diff = f['f_down_diff'][...][:, :, 1]*1000
            f_up = f['f_up'][...][:, :, 1]*1000
        
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
        'sat_cwp': sat_cwp_list,
        'sat_cer': sat_cer_list,
        'sat_f_down': sat_f_down_list,
        'sat_f_down_dir': sat_f_down_dir_list,
        'sat_f_down_diff': sat_f_down_diff_list,
        'sat_f_up': sat_f_up_list,
        'manual_cot': manual_cot_list,
        'manual_cth': manual_cth_list,
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
    
    # print("isinstance(sat_f_down_list, list):", isinstance(sat_f_down_list, list))
    
    # for cloud_list in [sat_cot_list, sat_cth_list, sat_cwp_list, sat_cer_list,
    #                    manual_cot_list, manual_cth_list, manual_cwp_list, manual_cer_list]:
    #     cloud_list = np.array(cloud_list)
    #     cloud_list[cloud_list<0] = np.nan  # set negative values to NaN
    
    # for ind, cloud_list in enumerate([date_list, alt_list, sza_list, seg_list,
    #                                 sat_cot_list, sat_cth_list, sat_cwp_list, sat_cer_list,
    #                                 manual_cot_list, manual_cth_list, manual_cwp_list, manual_cer_list,
    #                                 sat_f_down_list, sat_f_down_dir_list, sat_f_down_diff_list, sat_f_up_list,
    #                                 manual_f_down_list, manual_f_down_dir_list, manual_f_down_diff_list, manual_f_up_list,
    #                                 clear_f_down_list, clear_f_down_dir_list, clear_f_down_diff_list, clear_f_up_list]):
    #     print(f"Processing cloud list {ind} with length: {len(cloud_list)}")
    #     if isinstance(cloud_list, list):
    #         cloud_list = np.array(cloud_list)
    #         print("isinstance(sat_f_down_list, list):", isinstance(sat_f_down_list, list))
    #         print("isinstance(cloud_list, list):", isinstance(cloud_list, list))
    #     cloud_list = cloud_list[~sat_cot_nan_mask]
        
    # print("isinstance(sat_f_down_list, list):", isinstance(sat_f_down_list, list))
    
    # calculate the CRE at TOA
    # F_sat_cloud = sat_f_down_list - sat_f_up_list
    # F_manual_cloud = manual_f_down_list - manual_f_up_list
    # F_clear = clear_f_down_list - clear_f_up_list
    
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
    
    # plot the results
    fig = plt.figure(figsize=(5, 3.1))
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
        print("CRE sat:", F_cre_sat_cloud[mask])
        print("CRE manual:", F_cre_manual_cloud[mask])
        
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
    ax1.set_ylim(-122, 22)
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
               'date_postion_list': date_postion_list}
    
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
    
    fig = plt.figure(figsize=(5, 3.1))
    ax1 = fig.add_subplot(111)
    
    for i in range(len(sw_result['date_label_list'])):
        
        
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
    ax1.set_ylim(-122, 22)
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
    ax1.set_ylabel('Net CRE (W m$^{-2}$)', fontsize=label_size)
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
    fig = plt.figure(figsize=(5, 3.1))
    ax1 = fig.add_subplot(111)
    
    color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
    for ind, j in enumerate(sorted(set(date_list))):
        mask = np.array(df['date']) == j
        print(f"Processing date: {j}, mask length: {np.sum(mask)}")

        mm = np.int(j[4:6])
        dd = np.int(j[6:8])
        date_label = f"{mm:d}/{dd:d}"
        
        print()
        
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
    ax1.set_ylim(-122, 22)
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


def hypothetical_cre_multi(fnames_sw, fnames_lw,
                          fig_tag='test',
                          overwrite=False
                            ):


    zpt_list = []
    alb_list = []
    sza_list = []
    cot_list = []
    cth_list = []
    cbh_list = []
    cwp_list = []
    cer_list = []
    f_down_sw_list = []
    f_up_sw_list = []
    f_down_lw_list = []
    f_up_lw_list = []
    
    dwvl_nm = 1
    
    sw_z_ind = 0
    for fname_sw, fname_lw in zip(fnames_sw, fnames_lw):
        with h5py.File(fname_sw, 'r') as f:
            f_zpt_sw = f['zpt'][...]
            f_alb_sw = f['alb'][...]
            sza = f['sza'][...]
            f_sza = np.repeat(sza, f_zpt_sw.shape[0])  # repeat SZA for each zpt
            f_down_1d_sw = f['f_down'][...][:, :, sw_z_ind]
            # f_down_dir_1d_sw = f['f_down_dir'][...][:, :, sw_z_ind]
            # f_down_diff_1d_sw = f['f_down_diff'][...][:, :, sw_z_ind]
            f_up_1d_sw = f['f_up'][...][:, :, sw_z_ind]
            f_cth_sw = f['cth'][...]
            f_cbh_sw = f['cbh'][...]
            f_cot_sw = f['cot'][...]
            f_cwp_sw = f['cwp'][...] * 1000  # convert to g m^-2
            f_cer_sw = f['cer'][...]
            
            f_down_sw = np.sum(f_down_1d_sw*dwvl_nm, axis=1)  # sum over wavelengths
            f_up_sw = np.sum(f_up_1d_sw*dwvl_nm, axis=1)
            
            # print("f_down_1d_sw shape:", f_down_1d_sw.shape)
            # print("f_down_sw shape:", f_down_sw.shape)
            
            zpt_list.extend(f_zpt_sw)
            alb_list.extend(f_alb_sw)
            sza_list.extend(f_sza)
            cot_list.extend(f_cot_sw)
            cth_list.extend(f_cth_sw)
            cbh_list.extend(f_cbh_sw)
            cwp_list.extend(f_cwp_sw)
            cer_list.extend(f_cer_sw)
            f_down_sw_list.extend(f_down_sw)
            f_up_sw_list.extend(f_up_sw)

            
            
        
        # lw_out = pd.read_csv('output_0000.txt', sep='\s+', header=None)
        # lw_wvl = lw_out.iloc[:, 0].values.astype(float)  # first col is wavelength
        # dlw_wvl_nm = lw_wvl[1:] - lw_wvl[:-1]  # wavelength step in nm
        # dlw_wvl_nm = np.concatenate((np.array([dlw_wvl_nm[0]]), dlw_wvl_nm))  # add a zero for the last wavelength
        
        with h5py.File(fname_lw, 'r') as f:
            # f_zpt_lw = f['zpt'][...]
            # f_alb_lw = f['alb'][...]
            # f_sza_lw = f['sza'][...]
            f_down_1d_lw = f['f_down'][...][:, 0, 0]*1000
            # f_down_dir_lw = f['f_down_dir'][...][:, 0, 0]*1000
            # f_down_diff_lw = f['f_down_diff'][...][:, 0, 0]*1000
            f_up_1d_lw = f['f_up'][...][:, 0, 0]*1000
            # f_down_1d_lw = f['f_down'][...][:, :]*1000
            # f_down_dir_lw = f['f_down_dir'][...][:, :]*1000
            # f_down_diff_lw = f['f_down_diff'][...][:, :]*1000
            # f_up_1d_lw = f['f_up'][...][:, :]*1000
            # f_cth_lw = f['cth'][...]
            # f_cbh_lw = f['cbh'][...]
            # f_cot_lw = f['cot'][...]
            # f_cwp_lw = f['cwp'][...]  * 1000  # convert to g m^-2
            # f_cer_lw = f['cer'][...]
        
            # print("f['f_down'][...] shape:", f['f_down'][...].shape)
            # print("f_down_1d_lw shape:", f_down_1d_lw.shape)
            # sys.exit()
            # f_down_lw = np.sum(f_down_1d_lw*dlw_wvl_nm, axis=1)  # sum over wavelengths
            # f_up_lw = np.sum(f_up_1d_lw*dlw_wvl_nm, axis=1)
            
            f_down_lw = f_down_1d_lw
            f_up_lw = f_up_1d_lw
            
            f_down_lw_list.extend(f_down_lw)
            f_up_lw_list.extend(f_up_lw)
    

    zpt_set = sorted(set(zpt_list))
    alb_set = sorted(set(alb_list))
    sza_set = sorted(set(sza_list))
    cer_set = sorted(set(cer_list))
    
    # convert lists to a pandas DataFrame
    df = pd.DataFrame({
        'zpt': zpt_list,
        'alb': alb_list,
        'sza': sza_list,
        'cot': cot_list,
        'cth': cth_list,
        'cbh': cbh_list,
        'cwp': cwp_list,
        'cer': cer_list,
        'f_down_sw': f_down_sw_list,
        'f_up_sw': f_up_sw_list,
        'f_down_lw': f_down_lw_list,
        'f_up_lw': f_up_lw_list
    })
    
    df['cgt'] = (df['cth'] - df['cbh']).apply(lambda x: np.round(x, 1))  # calculate cloud geometric thickness
    
    df['cth_cgt'] = df['cth'].astype(str) + '_' + df['cgt'].astype(str)
    
    cth_cgt_set = sorted(set(df['cth_cgt']))
    
    
    # for cth_cgt in cth_cgt_set:
    #     mask = df['cth_cgt'] == cth_cgt
    #     print(f"Processing CTH_CGT: {cth_cgt}, mask length: {np.sum(mask)}")
        
    #     df_tmp = df[mask]
    #     df_tmp_grouped = df_tmp.groupby(['cwp'])
        
    #     cth_tmp = df_tmp['cth'].values[0]
    #     cgt_tmp = df_tmp['cgt'].values[0]
    #     cot_tmp = df_tmp['cot'].values
    #     cwp_tmp = df_tmp['cwp'].values
    #     cer_tmp = df_tmp['cer'].values
        
    #     clear_mask = cot_tmp == 0.0  # clear sky mask based on COT
        
    #     f_down_sw_clear = df_tmp['f_down_sw'][clear_mask]
        
    #     print("f_down_sw_clear:", f_down_sw_clear)
    #     sys.exit()
        
        
        
    #     f_up_sw_clear = df_tmp['f_up_sw'].values
    #     f_down_sw_cloud = df_tmp['f_down_sw']
    
    print("zpt_set:", zpt_set)
    print("alb_set:", alb_set)
    print("df[alb]=='sfc_alb_20240531.dat':", sum(df['alb']==b'sfc_alb_20240531.dat'))
    # sys.exit()
    
    df_mid_alb = df[df['alb'] == b'sfc_alb_20240531.dat']
    
    df_high_alb = df[df['alb'] == b'sfc_alb_20240607.dat']
    
    print('cth_cgt', set(df_mid_alb['cth_cgt']))
    print('cer', set(df_mid_alb['cer']))
    
    # '0.0_0.0', '0.2_0.1', '0.1_0.1', '3.2_0.2', '0.3_0.2', '1.2_0.2'
    
    df_mid_clear = df_mid_alb[df_mid_alb['cot'] == 0.0]
    df_mid_cth_cgt_02_01 = df_mid_alb[df_mid_alb['cth_cgt']== '0.2_0.1']
    df_mid_cth_cgt_01_01 = df_mid_alb[df_mid_alb['cth_cgt']== '0.1_0.1']
    df_mid_cth_cgt_32_02 = df_mid_alb[df_mid_alb['cth_cgt']== '3.2_0.2']
    df_mid_cth_cgt_03_02 = df_mid_alb[df_mid_alb['cth_cgt']== '0.3_0.2']
    df_mid_cth_cgt_12_02 = df_mid_alb[df_mid_alb['cth_cgt']== '1.2_0.2']
    
    # print("df_mid_clear shape:", df_mid_clear.shape)
    # print("df_mid_cth_cgt_02_01 shape:", df_mid_cth_cgt_02_01.shape)
    # print("df_mid_clear:", df_mid_clear)
    # print("df_mid_cth_cgt_02_01:", df_mid_cth_cgt_02_01)
    # print("df_mid_cth_cgt_02_01 col:", df_mid_cth_cgt_02_01.columns)    
    # # sys.exit()
    # sys.exit()
    
    df_high_clear = df_high_alb[df_high_alb['cot'] == 0.0]
    df_high_cth_cgt_02_01 = df_high_alb[df_high_alb['cth_cgt']== '0.2_0.1']
    df_high_cth_cgt_01_01 = df_high_alb[df_high_alb['cth_cgt']== '0.1_0.1']
    df_high_cth_cgt_32_02 = df_high_alb[df_high_alb['cth_cgt']== '3.2_0.2']
    df_high_cth_cgt_03_02 = df_high_alb[df_high_alb['cth_cgt']== '0.3_0.2']
    df_high_cth_cgt_12_02 = df_high_alb[df_high_alb['cth_cgt']== '1.2_0.2']
    
    
    print("df_high_clear shape:", df_high_clear.shape)
    print("df_high_cth_cgt_02_01 shape:", df_high_cth_cgt_02_01.shape)
    # sys.exit()
    
    # for zpt_file in zpt_set:
    #     for alb
        
    F_clear_sw_mid = np.float(df_mid_clear['f_down_sw'] - df_mid_clear['f_up_sw'])
    F_cloud_sw_mid_cth_cgt_02_01 = df_mid_cth_cgt_02_01['f_down_sw'] - df_mid_cth_cgt_02_01['f_up_sw']
    F_cloud_sw_mid_cth_cgt_01_01 = df_mid_cth_cgt_01_01['f_down_sw'] - df_mid_cth_cgt_01_01['f_up_sw']
    F_cloud_sw_mid_cth_cgt_32_02 = df_mid_cth_cgt_32_02['f_down_sw'] - df_mid_cth_cgt_32_02['f_up_sw']
    F_cloud_sw_mid_cth_cgt_03_02 = df_mid_cth_cgt_03_02['f_down_sw'] - df_mid_cth_cgt_03_02['f_up_sw']
    F_cloud_sw_mid_cth_cgt_12_02 = df_mid_cth_cgt_12_02['f_down_sw'] - df_mid_cth_cgt_12_02['f_up_sw']
    F_clear_lw_mid = np.float(df_mid_clear['f_down_lw'] - df_mid_clear['f_up_lw'])
    F_cloud_lw_mid_cth_cgt_02_01 = df_mid_cth_cgt_02_01['f_down_lw'] - df_mid_cth_cgt_02_01['f_up_lw']
    F_cloud_lw_mid_cth_cgt_01_01 = df_mid_cth_cgt_01_01['f_down_lw'] - df_mid_cth_cgt_01_01['f_up_lw']
    F_cloud_lw_mid_cth_cgt_32_02 = df_mid_cth_cgt_32_02['f_down_lw'] - df_mid_cth_cgt_32_02['f_up_lw']
    F_cloud_lw_mid_cth_cgt_03_02 = df_mid_cth_cgt_03_02['f_down_lw'] - df_mid_cth_cgt_03_02['f_up_lw']
    F_cloud_lw_mid_cth_cgt_12_02 = df_mid_cth_cgt_12_02['f_down_lw'] - df_mid_cth_cgt_12_02['f_up_lw']
    
    swCRE_mid_cth_cgt_02_01 = F_cloud_sw_mid_cth_cgt_02_01 - F_clear_sw_mid
    swCRE_mid_cth_cgt_01_01 = F_cloud_sw_mid_cth_cgt_01_01 - F_clear_sw_mid
    swCRE_mid_cth_cgt_32_02 = F_cloud_sw_mid_cth_cgt_32_02 - F_clear_sw_mid
    swCRE_mid_cth_cgt_03_02 = F_cloud_sw_mid_cth_cgt_03_02 - F_clear_sw_mid
    swCRE_mid_cth_cgt_12_02 = F_cloud_sw_mid_cth_cgt_12_02 - F_clear_sw_mid
    lwCRE_mid_cth_cgt_02_01 = F_cloud_lw_mid_cth_cgt_02_01 - F_clear_lw_mid
    lwCRE_mid_cth_cgt_01_01 = F_cloud_lw_mid_cth_cgt_01_01 - F_clear_lw_mid
    lwCRE_mid_cth_cgt_32_02 = F_cloud_lw_mid_cth_cgt_32_02 - F_clear_lw_mid
    lwCRE_mid_cth_cgt_03_02 = F_cloud_lw_mid_cth_cgt_03_02 - F_clear_lw_mid
    lwCRE_mid_cth_cgt_12_02 = F_cloud_lw_mid_cth_cgt_12_02 - F_clear_lw_mid
    
    F_net_mid_cth_cgt_02_01 = swCRE_mid_cth_cgt_02_01 + lwCRE_mid_cth_cgt_02_01
    F_net_mid_cth_cgt_01_01 = swCRE_mid_cth_cgt_01_01 + lwCRE_mid_cth_cgt_01_01
    F_net_mid_cth_cgt_32_02 = swCRE_mid_cth_cgt_32_02 + lwCRE_mid_cth_cgt_32_02
    F_net_mid_cth_cgt_03_02 = swCRE_mid_cth_cgt_03_02 + lwCRE_mid_cth_cgt_03_02
    F_net_mid_cth_cgt_12_02 = swCRE_mid_cth_cgt_12_02 + lwCRE_mid_cth_cgt_12_02
    
    F_clear_sw_high = np.float(df_high_clear['f_down_sw'] - df_high_clear['f_up_sw'])
    F_cloud_sw_high_cth_cgt_02_01 = df_high_cth_cgt_02_01['f_down_sw'] - df_high_cth_cgt_02_01['f_up_sw']
    F_cloud_sw_high_cth_cgt_01_01 = df_high_cth_cgt_01_01['f_down_sw'] - df_high_cth_cgt_01_01['f_up_sw']
    F_cloud_sw_high_cth_cgt_32_02 = df_high_cth_cgt_32_02['f_down_sw'] - df_high_cth_cgt_32_02['f_up_sw']
    F_cloud_sw_high_cth_cgt_03_02 = df_high_cth_cgt_03_02['f_down_sw'] - df_high_cth_cgt_03_02['f_up_sw']
    F_cloud_sw_high_cth_cgt_12_02 = df_high_cth_cgt_12_02['f_down_sw'] - df_high_cth_cgt_12_02['f_up_sw']
    F_clear_lw_high = np.float(df_high_clear['f_down_lw'] - df_high_clear['f_up_lw'])
    F_cloud_lw_high_cth_cgt_02_01 = df_high_cth_cgt_02_01['f_down_lw'] - df_high_cth_cgt_02_01['f_up_lw']
    F_cloud_lw_high_cth_cgt_01_01 = df_high_cth_cgt_01_01['f_down_lw'] - df_high_cth_cgt_01_01['f_up_lw']
    F_cloud_lw_high_cth_cgt_32_02 = df_high_cth_cgt_32_02['f_down_lw'] - df_high_cth_cgt_32_02['f_up_lw']
    F_cloud_lw_high_cth_cgt_03_02 = df_high_cth_cgt_03_02['f_down_lw'] - df_high_cth_cgt_03_02['f_up_lw']
    F_cloud_lw_high_cth_cgt_12_02 = df_high_cth_cgt_12_02['f_down_lw'] - df_high_cth_cgt_12_02['f_up_lw']   
    
    # print("df_high_cth_cgt_32_02['f_down_sw'] shape:", df_high_cth_cgt_32_02['f_down_sw'].shape)
    # print("df_high_clear['f_down_sw'] shape:", df_high_clear['f_down_sw'].shape)
    # print("F_clear_sw_high shape:", F_clear_sw_high.shape)
    # print("F_cloud_sw_high_cth_cgt_32_02 shape:", F_cloud_sw_high_cth_cgt_32_02.shape)
    # print("df_high_cth_cgt_02_01['f_down_sw'] - df_high_cth_cgt_02_01['f_up_sw'] shape:", 
    #       (df_high_cth_cgt_02_01['f_down_sw'] - df_high_cth_cgt_02_01['f_up_sw']).shape)
    # print("F_cloud_sw_high_cth_cgt_32_02 shape:", F_cloud_sw_high_cth_cgt_32_02.shape)
    
    swCRE_high_cth_cgt_02_01 = F_cloud_sw_high_cth_cgt_02_01 - F_clear_sw_high
    swCRE_high_cth_cgt_01_01 = F_cloud_sw_high_cth_cgt_01_01 - F_clear_sw_high
    swCRE_high_cth_cgt_32_02 = F_cloud_sw_high_cth_cgt_32_02 - F_clear_sw_high
    swCRE_high_cth_cgt_03_02 = F_cloud_sw_high_cth_cgt_03_02 - F_clear_sw_high
    swCRE_high_cth_cgt_12_02 = F_cloud_sw_high_cth_cgt_12_02 - F_clear_sw_high
    lwCRE_high_cth_cgt_02_01 = F_cloud_lw_high_cth_cgt_02_01 - F_clear_lw_high
    lwCRE_high_cth_cgt_01_01 = F_cloud_lw_high_cth_cgt_01_01 - F_clear_lw_high
    lwCRE_high_cth_cgt_32_02 = F_cloud_lw_high_cth_cgt_32_02 - F_clear_lw_high
    lwCRE_high_cth_cgt_03_02 = F_cloud_lw_high_cth_cgt_03_02 - F_clear_lw_high
    lwCRE_high_cth_cgt_12_02 = F_cloud_lw_high_cth_cgt_12_02 - F_clear_lw_high
    F_net_high_cth_cgt_02_01 = swCRE_high_cth_cgt_02_01 + lwCRE_high_cth_cgt_02_01
    F_net_high_cth_cgt_01_01 = swCRE_high_cth_cgt_01_01 + lwCRE_high_cth_cgt_01_01
    F_net_high_cth_cgt_32_02 = swCRE_high_cth_cgt_32_02 + lwCRE_high_cth_cgt_32_02
    F_net_high_cth_cgt_03_02 = swCRE_high_cth_cgt_03_02 + lwCRE_high_cth_cgt_03_02
    F_net_high_cth_cgt_12_02 = swCRE_high_cth_cgt_12_02 + lwCRE_high_cth_cgt_12_02

    
    # print("F_cloud_sw_high_cth_cgt_03_02 shape:", F_cloud_sw_high_cth_cgt_03_02.shape)
    # print("F_clear_sw_high shape:", F_clear_sw_high.shape)
    # print("F_cloud_sw_high_cth_cgt_03_02 - F_clear_sw_high shape:", 
    #       (F_cloud_sw_high_cth_cgt_03_02 - F_clear_sw_high).shape)
    # print("F_net_high_cth_cgt_02_01 shape:", F_net_high_cth_cgt_02_01.shape)
    # print("F_net_high_cth_cgt_01_01 shape:", F_net_high_cth_cgt_01_01.shape)
    # print("F_net_high_cth_cgt_32_02 shape:", F_net_high_cth_cgt_32_02.shape)
    # print("F_net_high_cth_cgt_03_02 shape:", F_net_high_cth_cgt_03_02.shape)
    # print("F_net_high_cth_cgt_12_02 shape:", F_net_high_cth_cgt_12_02.shape)

    # F_net = swCRE + lw CRE
    
    cwp_arr = np.array(df_mid_cth_cgt_02_01['cwp']).flatten()  # CWP for cloudy pixels
    # swCRE = np.array(swCRE).flatten()
    # lwCRE = np.array(lwCRE).flatten()
    # F_net = np.array(F_net).flatten()

    # cer_x = np.max(f_cer_lw) # CER for cloudy pixels
    # cth_x = np.max(f_cth_lw) # CTH for cloudy pixels
    # cbh_x = np.max(f_cbh_lw) # CBH for cloudy pixels
    # cgt_x = cth_x-cbh_x
    
    fig = plt.figure(figsize=(7, 5))
    ax1 = fig.add_subplot(111)
    ax1.plot(cwp_arr, swCRE_high_cth_cgt_03_02, label='SW higher albedo',)
    ax1.plot(cwp_arr, lwCRE_high_cth_cgt_03_02, label='LW higher albedo',)
    ax1.plot(cwp_arr, F_net_high_cth_cgt_03_02, label='higher albedo', color='tab:red')
    ax1.plot(cwp_arr, F_net_mid_cth_cgt_03_02, label='medium albedo', color='tab:blue')
    ax1.set_xlabel('CWP (g m$^{-2}$)', fontsize=label_size)
    ax1.set_ylabel('Net CRE (W m$^{-2}$)', fontsize=label_size)
    ax1.set_title(f'Hypothetical Cloud Radiative Effect\nCER 8 $\mu$m, CTH 0.3 km, CGT 200 m ',
                  fontsize=label_size+2)
    ax1.tick_params('both', labelsize=label_size-2)
    ax1.legend(loc='best', fontsize=legend_size)
    ax1.axhline(0, color='grey', linestyle='--', linewidth=0.5)
    fig.tight_layout()
    fig.savefig(f"fig/clouds/hypothetical_CRE_{fig_tag}_alb_03_02.png", dpi=300, bbox_inches='tight')
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
    fig = plt.figure(figsize=(5, 3.1))
    ax1 = fig.add_subplot(111)
    
    color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
    for ind, j in enumerate(sorted(set(date_list))):
        mask = np.array(df['date']) == j
        print(f"Processing date: {j}, mask length: {np.sum(mask)}")

        mm = np.int(j[4:6])
        dd = np.int(j[6:8])
        date_label = f"{mm:d}/{dd:d}"
        
        print()
        
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
    ax1.set_ylim(-122, 22)
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


if __name__ == '__main__':

   
    
    """
    sw_result = flt_trk_cre_2lay_sw(dates=['20240603', '20240607', '20240606', '20240613'],
                          case_tag=['cloudy_track_1', 'cloudy_track_2', 'cloudy_track_2', 'cloudy_track_1'],
                          separate_heights=[600, 300, 300, 300],
                          fig_tag='test',
                          overwrite=False
                            )
    
    lw_result = flt_trk_cre_2lay_lw(dates=['20240603', '20240607', '20240606', '20240613'],
                          case_tag=['cloudy_track_1', 'cloudy_track_2', 'cloudy_track_2', 'cloudy_track_1'],
                          separate_heights=[600, 300, 300, 300],
                          fig_tag='test',
                          overwrite=False
                            )
    
    plot_combine_results(sw_result, lw_result, fig_tag='test')
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
    
    # hypothetical_cre('flt_trk_lrt_para-lrt-20240531-hypothetical_test_mid_cld_high_alb-hypothetical-sw.h5', 
    #                  'flt_trk_lrt_para-lrt-20240531-hypothetical_test_mid_cld_high_alb-hypothetical-lw.h5',
    #                       fig_tag='test',
    #                       overwrite=False
    #                         )
    
    hypothetical_cre_multi(['flt_trk_lrt_para-lrt-20240531-test_cld_alb_8um-hypothetical-sw.h5'], 
                            ['flt_trk_lrt_para-lrt-20240531-test_cld_alb_8um-hypothetical-sw.h5'],
                          fig_tag='test_multi',
                          overwrite=False
                            )