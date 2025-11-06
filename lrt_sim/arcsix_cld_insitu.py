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
from pathlib import Path
import logging
from typing import Optional
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
from scipy.interpolate import interp1d
import netCDF4 as nc
from netCDF4 import Dataset as NcDataset
import xarray as xr
from collections import defaultdict
import platform
# mpl.use('Agg')


import er3t

from util.util import *

_mission_      = 'arcsix'
_platform_     = 'p3b'

_hsk_          = 'hsk'
_alp_          = 'alp'
_spns_         = 'spns-a'
_ssfr1_        = 'ssfr-a'
_ssfr2_        = 'ssfr-b'
_cam_          = 'nac'

# _fdir_main_       = 'data/%s/flt-vid' % _mission_
_fdir_main_       = 'data/flt-vid'
_fdir_sat_img_    = 'data/%s/sat-img' % _mission_
_fdir_sat_data_   = 'data/%s/sat' % _mission_
_fdir_cam_img_    = 'data/%s/2024-Spring/p3' % _mission_
_wavelength_      = 555.0

_fdir_sat_img_vn_ = 'data/%s/sat-img-vn' % _mission_

_preferred_region_ = 'ca_archipelago'
_aspect_ = 'equal'

if platform.system() == 'Darwin':
    _fdir_data_ = '/Volumes/argus/field/%s/processed' % _mission_
    _fdir_data_ = 'data/processed' 
    _fdir_general_ = 'data'
    _fdir_tmp_ = './tmp'
elif platform.system() == 'Linux':
    _fdir_data_ = "/pl/active/vikas-arcsix/yuch8913/arcsix/data/processed"
    _fdir_general_ = "/pl/active/vikas-arcsix/yuch8913/arcsix/data"
    _fdir_tmp_ = "/pl/active/vikas-arcsix/yuch8913/arcsix/tmp"
_fdir_tmp_graph_ = 'tmp-graph_flt-vid'



def flt_trk_lwc(
        date: datetime.datetime = datetime.datetime(2024,6,5),
        tmhr_ranges_select = ((15.36,15.6),(16.32,16.6)),
        output_lwp_alt=[False, True],
        fname_LWC: str = "data/lwc/ARCSIX-Lwc123_P3B_20240611_R1.ict",
        fname_cloud_micro_2DGRAY50: str = "data/cloud_prob/ARCSIX-2DGRAY50_P3B_20240611105230_R1.ict",
        fname_cloud_micro_FCDP: str = "data/cloud_prob/ARCSIX-FCDP_P3B_20240611105230_R1.ict",
        timeoff_2DGRAY50: float = -0.0,  # hours
        timeoff_FCDP: float = -0.0,      # hours
        config: Optional[FlightConfig] = None,
    ):

    log = logging.getLogger("marli")
    log.info("Starting processing for %s", date.strftime("%Y%m%d"))

    assert config, "FlightConfig required"
    date_s = date.strftime("%Y%m%d")
    dir_fig = f'./fig/{date_s}'
    os.makedirs(dir_fig, exist_ok=True)

    # 1) Load all instrument & satellite metadata
    data_hsk  = load_h5(config.hsk(date_s))
    data_ssfr = load_h5(config.ssfr(date_s))
    data_hsr1 = load_h5(config.hsr1(date_s))
    head, data_lwc = read_ict_lwc(fname_LWC, encoding='utf-8', na_values=[-9999999, -777, -888])
    head, data_cloud_2DGRAY50 = read_ict_cloud_micro_2DGRAY50(fname_cloud_micro_2DGRAY50, encoding='utf-8', na_values=[-9999999, -777, -888])
    head, data_cloud_FCDP = read_ict_cloud_micro_FCDP(fname_cloud_micro_FCDP, encoding='utf-8', na_values=[-9999999, -777, -888])
        

    # Build leg masks
    t_hsk = np.array(data_hsk["tmhr"])
    leg_masks = [(t_hsk>=lo)&(t_hsk<=hi) for lo,hi in tmhr_ranges_select]
    
    t_ssfr = data_ssfr['time']/3600.0  # convert to hours
    t_hsr1 = data_hsr1['time']/3600.0  # convert to hours
    t_lwc = np.array(data_lwc['tmhr'])
    t_cm_2DGRAY50 = np.array(data_cloud_2DGRAY50['tmhr']) + timeoff_2DGRAY50
    t_cm_FCDP = np.array(data_cloud_FCDP['tmhr']) + timeoff_FCDP

    # --- LOOP LEGS ---
    for i, mask in enumerate(leg_masks):
        times_leg = t_hsk[mask]

        # find index arrays in one go
        sel_ssfr, sel_hsr1, sel_lwc, sel_cm_2DGRAY50, sel_cm_FCDP = (
            nearest_indices(t_hsk, mask, arr)
            for arr in (t_ssfr, t_hsr1, t_lwc, t_cm_2DGRAY50, t_cm_FCDP)
        )

        # assemble a small dict for this leg
        leg = {
            "time":    times_leg,
            "alt":     data_hsk["alt"][mask] / 1000.0,
            "twc":     np.array(data_lwc["TWC"][sel_lwc]),
            "lwc1":    np.array(data_lwc["LWC_1"][sel_lwc]),
            "lwc2":    np.array(data_lwc["LWC_2"][sel_lwc]),
            "hsr1_tot": data_hsr1["f_dn_tot"][sel_hsr1],
            "hsr1_dif": data_hsr1["f_dn_dif"][sel_hsr1],
            "ssfr_zen": data_ssfr["f_dn"][sel_ssfr],
        }
        # cloud‐micro fields
        leg["conc_fcdp"] = np.array(data_cloud_FCDP["conc"])[sel_cm_FCDP]
        leg["ext_fcdp"]  = np.array(data_cloud_FCDP["ext"])[sel_cm_FCDP]
        leg["lwc_fcdp"] = np.array(data_cloud_FCDP["lwc"])[sel_cm_FCDP]
        
        leg["conc_2DGRAY50"] = np.array(data_cloud_2DGRAY50["conc"])[sel_cm_2DGRAY50]
        leg["ext_2DGRAY50"]  = np.array(data_cloud_2DGRAY50["ext"])[sel_cm_2DGRAY50]
        leg["iwc_2DGRAY50"] = np.array(data_cloud_2DGRAY50["iwc"])[sel_cm_2DGRAY50]
        leg["cer_2DGRAY50"] = np.array(data_cloud_2DGRAY50["effectiveDiam"])[sel_cm_2DGRAY50]
            

        # --- FIND THRESHOLD CROSSINGS ---
        thr = 0.005  # TWC threshold
        above = (leg["lwc1"] > thr).astype(int)
        crossings = np.where(np.diff(above))[0]
        if crossings.size:
            start, end = crossings[0], crossings[-1]
            log.info("Leg %d crossing alt: %.3f–%.3f km",
                     i, leg["alt"][start], leg["alt"][end])

        # --- PLOT TIME SERIES PANEL ---
        fig, axes = plt.subplots(2, 3, figsize=(18,8))
        ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
        # plt.rcParams['font.size'] = 12  # Sets default font size for all text elements
        plt.rcParams['axes.titlesize'] = 16 # Sets default font size for titles
        # plt.rcParams['axes.labelsize'] = 12 # Sets default font size for axis labels
        
        
        # altitude
        ax1.plot(leg["time"], leg["alt"], "k.-")
        ax1.set(title="P3B Altitude", xlabel="Time (UTC)", ylabel="Altitude (km)")

        # LWC
        ax2.plot(leg["time"], leg["twc"], "b.-", label="TWC")
        ax2.plot(leg["time"], leg["lwc1"], "k.-", label="LWC1")
        ax2.plot(leg["time"], leg["lwc2"], "r.-", label="LWC2")
        ax2.plot(leg["time"], leg["lwc_fcdp"], "g.-", label="FCDP LWC")
        ax2.legend(); ax2.set(title="P3B LWC", xlabel="Time (UTC)", ylabel="LWC (g m$^{-3}$)")
        if crossings.size:
            for ax in [ax1, ax2]:
                ax.axvline(leg["time"][crossings[0]], color="gray", ls="--", lw=0.5, alpha=0.5)
                ax.axvline(leg["time"][crossings[-1]], color="gray", ls="--", lw=0.5, alpha=0.5)

        # HSR1 diff ratio at 550-nm
        wvl=550.0
        idx_hsr1 = np.argmin(abs(data_hsr1["wvl_dn_tot"] - wvl))
        dif_ratio = leg["hsr1_dif"][:,idx_hsr1] / leg["hsr1_tot"][:,idx_hsr1]
        ax3.plot(leg["time"], dif_ratio, "k.-")
        ax3.set(title=f"HSR1 Diff Ratio @ {wvl:.0f}nm", xlabel="Time (UTC)", ylabel="HSR1 diffussion ratio")
        
        # SSFR vs HSR1 flux
        idx_ssfr = np.argmin(abs(data_ssfr["wvl_dn"] - wvl))
        ax4.plot(leg["time"], leg["ssfr_zen"][:,idx_ssfr], "k.-", label="SSFR zen")
        ax4.plot(leg["time"], leg["hsr1_tot"][:,idx_hsr1], "r.-", label="HSR1 tot")
        ax4.legend()
        ax4.set(
            title=f"SSFR Downward and HSR1 total Flux @ {wvl:.0f}nm", xlabel="UTC hr", ylabel="Flux (W m$^{-2}$)"
        )

        # cloud micro physics
        ax5.plot(leg["time"], leg["ext_fcdp"], "k.-")
        ax5.set(title="Cloud Extinction", xlabel="Time (UTC)", ylabel="FCDP Cloud extinction (km^${-1}$)")


        ax6.plot(leg["time"], leg["iwc_2DGRAY50"], "b.-", label="2DGRAY50 IWC")
        ax6.plot(leg["time"], leg["lwc_fcdp"], "k.-", label="FCDP LWC")
        ax6.set(title="FCDP LWC", xlabel="Time (UTC)", ylabel="Water content (g/m^${-3}$)")
        ax6.legend()

        fig.suptitle(f'P3B LWC and HSR1/SSFR data for {date_s} - {times_leg[0]:.2f} to {times_leg[-1]:.2f}', fontsize=20)
        fig.tight_layout(rect=[0,0,1,1])
        fig.savefig(f"{dir_fig}/P3B_LWC_HSR1_SSFR_{date_s}_{times_leg[0]:.2f}_{times_leg[-1]:.2f}.png")
        plt.close(fig)

        # --- OPTIONAL LWP / COT / CER CALC AND PLOT ---
        if output_lwp_alt[i] and crossings.size:
            zs = leg["alt"][start:end+1]
            lwc_prof = leg["lwc1"][start:end+1]
            lwc_fcdp_prof = leg["lwc_fcdp"][start:end+1]
            iwc_prof = leg["iwc_2DGRAY50"][start:end+1]
            ext_prof = leg["ext_fcdp"][start:end+1]
            ext_ice_prof = leg["ext_2DGRAY50"][start:end+1]
            

            # ensure sorted by altitude
            order = np.argsort(zs)
            zs, lwc_prof, lwc_fcdp_prof, ext_prof = zs[order], lwc_prof[order], lwc_fcdp_prof[order], ext_prof[order]
            iwc_prof, ext_ice_prof = iwc_prof[order], ext_ice_prof[order]

            # dz = 0.002  # km
            # z_grid = np.arange(zs.min(), zs.max()+dz, dz)
            # lwc_i = interp1d(zs, lwc_prof, bounds_error=False, fill_value=np.nan)(z_grid)
            # ext_i = interp1d(zs, ext_prof, bounds_error=False, fill_value=np.nan)(z_grid)
            
 
            dz = 0.002  # km
            z_min, z_max = zs.min(), zs.max()
            # define bin edges so that each grid center is mid‐bin
            bin_edges = np.arange(z_min - dz/2, z_max + dz, dz)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            z_grid = bin_centers
            counts, _    = np.histogram(zs, bins=bin_edges)
            sum_lwc, _   = np.histogram(zs, bins=bin_edges, weights=lwc_prof)
            sum_ext, _   = np.histogram(zs, bins=bin_edges, weights=ext_prof)
            sum_iwc, _   = np.histogram(zs, bins=bin_edges, weights=iwc_prof)
            sum_ext_ice, _ = np.histogram(zs, bins=bin_edges, weights=ext_ice_prof)
            sum_lwc_FCDP, _ = np.histogram(zs, bins=bin_edges, weights=lwc_fcdp_prof)
            # avoid division by zero
            with np.errstate(divide="ignore", invalid="ignore"):
                lwc_binned = sum_lwc / counts
                ext_binned = sum_ext / counts
                iwc_binned = sum_iwc / counts
                ext_ice_binned = sum_ext_ice / counts
                lwc_FCDP_binned = sum_lwc_FCDP / counts

            mask_lwc = counts>0
            mask_ext = counts>0
            mask_iwc = counts>0
            mask_ext_ice = counts>0
            mask_lwc_FCDP = counts>0

            lwc_i = interp1d(
                bin_centers[mask_lwc], lwc_binned[mask_lwc],
                bounds_error=False, fill_value=np.nan
            )(bin_centers)

            ext_i = interp1d(
                bin_centers[mask_ext], ext_binned[mask_ext],
                bounds_error=False, fill_value=np.nan
            )(bin_centers)
            
            iwc_i = interp1d(
                bin_centers[mask_iwc], iwc_binned[mask_iwc],
                bounds_error=False, fill_value=np.nan
            )(bin_centers)
            
            ext_ice_i = interp1d(
                bin_centers[mask_ext_ice], ext_ice_binned[mask_ext_ice],
                bounds_error=False, fill_value=np.nan
            )(bin_centers)
            
            lwc_FCDP_i = interp1d(
                bin_centers[mask_lwc_FCDP], lwc_FCDP_binned[mask_lwc_FCDP],
                bounds_error=False, fill_value=np.nan
            )(bin_centers)

            



            # LWP (kg/m²) and COT
            lwp = np.nansum(lwc_i * dz * 1e3) / 1e3  
            cot = np.nansum(ext_i * dz)
            rho_w = 1e3  # kg/m³
            cer = (3/2 * lwp) / (cot * rho_w) * 1e6 if (lwp>0 and cot>0) else np.nan
            
            lwp_FCDP = np.nansum(lwc_FCDP_i * dz * 1000)/1000 # kg/m^2
            cot_FCDP = np.nansum(ext_i * dz)
            if lwp_FCDP > 0. and cot > 0.0:
                cer_FCDP = 3/2 * lwp_FCDP / (cot * rho_w) * 1e6 # um
            else:
                cer_FCDP = np.nan
            


            fig, (axl, axe) = plt.subplots(1,2,figsize=(12,6.5))
            axl.plot(lwc_prof, zs, "k.-", lw=1.0, markersize=2.0, label="LWC (all)", zorder=10)
            axl.plot(lwc_i, z_grid, "r-",  lw=3.0, markersize=2.0, label="LWC (interp)", zorder=50)
            axl.set(title=f"LWC vs Altitude", xlabel="LWC (g m$^{-3}$)", ylabel="Altitude (km)")
            
            text_fontsize = 10


            axl.plot(lwc_FCDP_i, z_grid, 'g-', lw=3.0, label='FCDP LWC (interp)', zorder=60)
            axl.plot(lwc_fcdp_prof, zs, 'o-', color='orange', lw=1.0, markersize=2.0, label='FCDP LWC (all)', zorder=20)        
            axl.text(0.6, 0.3, f'FCDP LWP: {lwp_FCDP*1000:.2f} '+'g m$^{-2}$', transform=axl.transAxes, fontsize=text_fontsize, va='top', ha='left')
            
            axl.plot(iwc_i, z_grid, 'b-', lw=2.0, label='2DGRAY50 IWC (interp)', zorder=30)
            # axl.plot(iwc_prof, zs, 'o-', color='cyan', lw=1.0, markersize=2.0, label='2DGRAY50 IWC (all)')

            axl.legend(loc='center left', fontsize=10, bbox_to_anchor=(0.025, -0.15), ncol=3)

            axl.text(0.6, 0.25, f'LWP: {lwp*1000:.2f} '+'g m$^{-2}$', transform=axl.transAxes, fontsize=text_fontsize, va='top', ha='left')
            # cloud altitude
            axl.text(0.6, 0.2, f'Alt: {z_grid.min():.3f} to {z_grid.max():.3f} km', transform=axl.transAxes, fontsize=text_fontsize, va='top', ha='left')

 

            axe.plot(ext_prof, zs, "k.-", lw=1.0, markersize=2.0, label='Extinction data (all)')
            axe.plot(ext_i, z_grid, "b-", lw=2.5, label='Extinction data (all)')
            axe.text(0.65, 0.25, f'COT: {cot:.3f}', transform=axe.transAxes, fontsize=text_fontsize, va='top', ha='left')
            axe.text(0.65, 0.2, f'CER: {cer:.1f} um', transform=axe.transAxes, fontsize=text_fontsize, va='top', ha='left')
            axe.text(0.65, 0.15, f'FCDP CER: {cer_FCDP:.1f} um', transform=axe.transAxes, fontsize=text_fontsize, va='top', ha='left')
            axe.set(title=f"Extinction", xlabel="Extinction (km$^{-1}$)", ylabel="Altitude (km)")

            fig.suptitle(f'P3B LWC and Cloud Microphysics for {date_s} - {times_leg[0]:.2f} to {times_leg[-1]:.2f}', fontsize=20)
            # fig.tight_layout(rect=[0,0,1,1])
            fig.tight_layout()
            fig.savefig(f'fig/{date_s}/P3B_LWP_vs_Altitude_{date_s}_{times_leg[0]:.2f}_{times_leg[-1]:.2f}.png', bbox_inches='tight', dpi=300)
            plt.close(fig)
            log.info("Leg %d LWP: %.4f g m$^{-2}$, COT: %.4f, CER: %.1f um", i, lwp*1000, cot, cer)
            log.info("Leg %d FCDP LWP: %.4f g m$^{-2}$, COT: %.4f, CER: %.1f um", i, lwp_FCDP*1000, cot_FCDP, cer_FCDP)
            
    return None
  

if __name__ == '__main__':
    
    dir_fig = './fig'
    os.makedirs(dir_fig, exist_ok=True)
    
    config = FlightConfig(mission='ARCSIX',
                            platform='P3B',
                            data_root=_fdir_data_,
                            sat_root_mac='/Volumes/argus/field/arcsix/sat-data',
                            sat_root_linux='/pl/active/vikas-arcsix/yuch8913/arcsix/data/sat-data',)
       
    
    # flt_trk_lwc(date=datetime.datetime(2024, 6, 5),
    #             tmhr_ranges_select=[[13, 17], [14.5, 16.1], [15.04, 15.28]],
    #             output_lwp_alt=[False, False, True],
    #             fname_LWC=f'{_fdir_general_}/lwc/ARCSIX-Lwc123_P3B_20240605_R1.ict',
    #             fname_cloud_micro_2DGRAY50=f'{_fdir_general_}/cloud_prob/2DGRAY50/ARCSIX-2DGRAY50_P3B_20240611105230_R1.ict',
    #             fname_cloud_micro_FCDP=f'{_fdir_general_}/cloud_prob/FCDP/ARCSIX-FCDP_P3B_20240605_R1.ict',
    #             timeoff_FCDP=0.005,
    #             config=config
    #             )
    
    flt_trk_lwc(date=datetime.datetime(2024, 6, 11),
                tmhr_ranges_select=[[13.9111, 15.7139], [14.03, 14.075], [16.268, 16.345]],
                output_lwp_alt=[False, True, True],
                fname_LWC=f'{_fdir_general_}/lwc/ARCSIX-Lwc123_P3B_20240611_R1.ict',
                fname_cloud_micro_2DGRAY50=f'{_fdir_general_}/cloud_prob/2DGRAY50/ARCSIX-2DGRAY50_P3B_20240611105230_R1.ict',
                fname_cloud_micro_FCDP=f'{_fdir_general_}/cloud_prob/FCDP/ARCSIX-FCDP_P3B_20240611_R1.ict',
                timeoff_FCDP=0.001,
                config=config
                )
    
    
    # flt_trk_lwc(date=datetime.datetime(2024, 6, 7),
    #             tmhr_ranges_select=[[15.34, 16.27], [15.765, 15.795]],
    #             output_lwp_alt=[False, True],
    #             fname_LWC=f'{_fdir_general_}/lwc/ARCSIX-Lwc123_P3B_20240607_R1.ict',
    #             fname_cloud_micro_2DGRAY50=f'{_fdir_general_}/cloud_prob/2DGRAY50/ARCSIX-2DGRAY50_P3B_20240607104243_R1.ict',
    #             fname_cloud_micro_FCDP=f'{_fdir_general_}/cloud_prob/FCDP/ARCSIX-FCDP_P3B_20240607_R1.ict',
    #             timeoff_FCDP=0.002,
    #             config=config
    #             )
    
    # flt_trk_lwc(date=datetime.datetime(2024, 6, 7),
    #             tmhr_ranges_select=[[15.19, 15.31], [15.34, 16.27], [15.765, 15.795]],
    #             output_lwp_alt=[True, False, True],
    #             fname_LWC=f'{_fdir_general_}/lwc/ARCSIX-Lwc123_P3B_20240607_R1.ict',
    #             fname_cloud_micro_2DGRAY50=f'{_fdir_general_}/cloud_prob/2DGRAY50/ARCSIX-2DGRAY50_P3B_20240607104243_R1.ict',
    #             fname_cloud_micro_FCDP=f'{_fdir_general_}/cloud_prob/FCDP/ARCSIX-FCDP_P3B_20240607_R1.ict',
    #             timeoff_FCDP=0.002,
    #             config=config
    #             )
    
    # flt_trk_lwc(date=datetime.datetime(2024, 6, 13),
    #             tmhr_ranges_select=[[14.0, 16.5], [14.92, 15.28], [15.03, 15.11], [15.781, 15.852], [15.8, 16.1], [15.88, 15.94], [16.06, 16.126]],
    #             output_lwp_alt=[False, False, True, True, False, True, True],
    #             fname_LWC=f'{_fdir_general_}/lwc/ARCSIX-Lwc123_P3B_20240613_R1.ict',
    #             fname_cloud_micro_2DGRAY50=f'{_fdir_general_}/cloud_prob/2DGRAY50/ARCSIX-2DGRAY50_P3B_20240611105230_R1.ict',
    #             fname_cloud_micro_FCDP=f'{_fdir_general_}/cloud_prob/FCDP/ARCSIX-FCDP_P3B_20240613_R1.ict',
    #             timeoff_FCDP=0.017,
    #             config=config
    #             )
    
    # flt_trk_lwc(date=datetime.datetime(2024, 6, 3),
    #             tmhr_ranges_select=[[14.51, 15.76], [14.84, 14.97], [13.23, 13.95], [13.39, 13.60], [13.73, 13.83]],
    #             output_lwp_alt=[False, True, False, True, False, True],
    #             fname_LWC=f'{_fdir_general_}/lwc/ARCSIX-Lwc123_P3B_20240603_R1.ict',
    #             fname_cloud_micro_2DGRAY50=f'{_fdir_general_}/cloud_prob/2DGRAY50/ARCSIX-2DGRAY50_P3B_20240611105230_R1.ict',
    #             fname_cloud_micro_FCDP=f'{_fdir_general_}/cloud_prob/FCDP/ARCSIX-FCDP_P3B_20240603_R1.ict',
    #             timeoff_FCDP=0.043,
    #             config=config
    #             )
    
    # flt_trk_lwc(date=datetime.datetime(2024, 6, 6),
    #             tmhr_ranges_select=[[13.48, 14.44], [13.63, 13.73], [13.76, 13.92]],
    #             output_lwp_alt=[True, True, True],
    #             fname_LWC=f'{_fdir_general_}/lwc/ARCSIX-Lwc123_P3B_20240606_R1.ict',
    #             fname_cloud_micro_2DGRAY50=f'{_fdir_general_}/cloud_prob/2DGRAY50/ARCSIX-2DGRAY50_P3B_20240611105230_R1.ict',
    #             fname_cloud_micro_FCDP=f'{_fdir_general_}/cloud_prob/FCDP/ARCSIX-FCDP_P3B_20240606_R1.ict',
    #             timeoff_FCDP=0.002,
    #             config=config
    #             )

    pass
