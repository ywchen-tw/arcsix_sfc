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
from dataclasses import dataclass
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

from util import gaussian, read_ict_radiosonde, read_ict_dropsonde, read_ict_lwc, read_ict_cloud_micro_2DGRAY50, read_ict_cloud_micro_FCDP, read_ict_bbr, read_ict_kt19
from util import read_ict_dlh_h2o, read_ict_co, read_ict_ch4, read_ict_co2, read_ict_o3

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

_title_extra_ = 'ARCSIX RF#1'

_tmhr_range_ = {
        '20240517': [19.20, 23.00],
        '20240521': [14.80, 17.50],
        }

# dates for ARCSIX-1
#╭────────────────────────────────────────────────────────────────────────────╮#
_dates1_ = [
        datetime.datetime(2024, 5, 28),
        datetime.datetime(2024, 5, 30),
        datetime.datetime(2024, 5, 31),
        datetime.datetime(2024, 6,  3),
        datetime.datetime(2024, 6,  5),
        datetime.datetime(2024, 6,  6),
        datetime.datetime(2024, 6,  7),
        datetime.datetime(2024, 6, 10),
        datetime.datetime(2024, 6, 11),
        datetime.datetime(2024, 6, 13),
    ]
#╰────────────────────────────────────────────────────────────────────────────╯#

# dates for ARCSIX-2
#╭────────────────────────────────────────────────────────────────────────────╮#
_dates2_ = [
        datetime.datetime(2024, 7, 25),
        datetime.datetime(2024, 7, 29),
        datetime.datetime(2024, 7, 30),
        datetime.datetime(2024, 8,  1),
        datetime.datetime(2024, 8,  2),
        datetime.datetime(2024, 8,  7),
        datetime.datetime(2024, 8,  8),
        datetime.datetime(2024, 8,  9),
        datetime.datetime(2024, 8,  15),
    ]
#╰────────────────────────────────────────────────────────────────────────────╯#

_dates_ = _dates1_


# --- Configuration ----------------------------------------------------------

@dataclass(frozen=True)
class FlightConfig:
    mission: str
    platform: str
    data_root: Path
    sat_root_mac: Path
    sat_root_linux: Path

    def hsk(self, date_s):    return f"{self.data_root}/{self.mission}-HSK_{self.platform}_{date_s}_v0.h5"
    def ssfr(self, date_s):   return f"{self.data_root}/{self.mission}-SSFR_{self.platform}_{date_s}_R0.h5"
    def hsr1(self, date_s):   return f"{self.data_root}/{self.mission}-HSR1_{self.platform}_{date_s}_R0.h5"
    def logic(self, date_s):  return f"{self.data_root}/{self.mission}-LOGIC_{self.platform}_{date_s}_RA.h5"
    def sat_coll(self, date_s): return f"{self.data_root}/{self.mission}-SAT-CLD_{self.platform}_{date_s}_v0.h5"
    def marli(self, fname):   return Path(fname)
    def kt19(self, fname):    return Path(fname)
    def sat_nc(self, date_s, raw):  # choose root by platform
        root = self.sat_root_mac if sys.platform=="darwin" else self.sat_root_linux
        return f"{root}/{date_s}/{raw}"

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
        thr = 0.008  # TWC threshold
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
            


            fig, (axl, axe) = plt.subplots(1,2,figsize=(12,6))
            axl.plot(lwc_prof, zs, "k.-", lw=1.0, markersize=2.0, label="LWC (all)", zorder=10)
            axl.plot(lwc_i, z_grid, "r-",  lw=3.0, markersize=2.0, label="LWC (interp)", zorder=50)
            axl.set(title=f"LWC vs Altitude", xlabel="LWC (g m$^{-3}$)", ylabel="Altitude (km)")
            
            text_fontsize = 10


            axl.plot(lwc_FCDP_i, z_grid, 'g-', lw=3.0, label='FCDP LWC (interp)', zorder=60)
            axl.plot(lwc_fcdp_prof, zs, 'o-', color='orange', lw=1.0, markersize=2.0, label='FCDP LWC (all)', zorder=20)        
            axl.text(0.65, 0.3, f'FCDP LWP: {lwp_FCDP*1000:.3f} '+'g m$^{-2}$', transform=axl.transAxes, fontsize=text_fontsize, va='top', ha='left')
            
            axl.plot(iwc_i, z_grid, 'b-', lw=2.0, label='2DGRAY50 IWC (interp)', zorder=30)
            # axl.plot(iwc_prof, zs, 'o-', color='cyan', lw=1.0, markersize=2.0, label='2DGRAY50 IWC (all)')

            axl.legend()

            axl.text(0.65, 0.25, f'LWP: {lwp*1000:.4f} '+'g m$^{-2}$', transform=axl.transAxes, fontsize=text_fontsize, va='top', ha='left')
            # cloud altitude
            axl.text(0.65, 0.2, f'Alt: {z_grid.min():.3f} to {z_grid.max():.3f} km', transform=axl.transAxes, fontsize=text_fontsize, va='top', ha='left')

 

            axe.plot(ext_prof, zs, "k.-", lw=1.0, markersize=2.0, label='Extinction data (all)')
            axe.plot(ext_i, z_grid, "b-", lw=2.5, label='Extinction data (all)')
            axe.text(0.7, 0.25, f'COT: {cot:.4f}', transform=axe.transAxes, fontsize=text_fontsize, va='top', ha='left')
            axe.text(0.7, 0.2, f'CER: {cer:.1f} um', transform=axe.transAxes, fontsize=text_fontsize, va='top', ha='left')
            axe.text(0.7, 0.15, f'FCDP CER: {cer_FCDP:.1f} um', transform=axe.transAxes, fontsize=text_fontsize, va='top', ha='left')
            axe.set(title=f"Extinction", xlabel="Extinction (km$^{-1}$)", ylabel="Altitude (km)")

            fig.suptitle(f'P3B LWC and Cloud Microphysics for {date_s} - {times_leg[0]:.2f} to {times_leg[-1]:.2f}', fontsize=20)
            # fig.tight_layout(rect=[0,0,1,1])
            fig.tight_layout()
            fig.savefig(f'fig/{date_s}/P3B_LWP_vs_Altitude_{date_s}_{times_leg[0]:.2f}_{times_leg[-1]:.2f}.png', bbox_inches='tight', dpi=300)
            plt.close(fig)
            log.info("Leg %d LWP: %.4f g m$^{-2}$, COT: %.4f, CER: %.1f um", i, lwp*1000, cot, cer)
            log.info("Leg %d FCDP LWP: %.4f g m$^{-2}$, COT: %.4f, CER: %.1f um", i, lwp_FCDP*1000, cot_FCDP, cer_FCDP)
            
    return None


def flt_trk_flux_R0(
        date: datetime.datetime = datetime.datetime(2024, 5, 31),
        tmhr_ranges_select = [[14.10, 14.27]],
        wvl_plot: float = 550.0,
        config: Optional[FlightConfig] = None,
    ):


    log = logging.getLogger("flt_trk_lwc")
    log.info("Starting processing for %s", date.strftime("%Y%m%d"))

    assert config, "FlightConfig required"
    date_s = date.strftime("%Y%m%d")
    dir_fig = f'./fig/{date_s}'
    os.makedirs(dir_fig, exist_ok=True)

    # 1) Load all instrument & satellite metadata
    data_hsk  = load_h5(config.hsk(date_s))
    data_ssfr = load_h5(config.ssfr(date_s))
    data_hsr1 = load_h5(config.hsr1(date_s))


    # Build leg masks
    t_hsk = np.array(data_hsk["tmhr"])
    leg_masks = [(t_hsk>=lo)&(t_hsk<=hi) for lo,hi in tmhr_ranges_select]
    
    t_ssfr = data_ssfr['time']/3600
    t_hsr1 = data_hsr1['tmhr']

    
    time_series_all = []
    vars()["cld_leg_all"] = {}
    
    # Loop legs: load raw NC, apply cloud logic, interpolate, plot
    for i, mask in enumerate(leg_masks):
        
        vars()[f"ssfr_logics_select_{i}_ind"] = []
        vars()[f"hsr1_logics_select_{i}_ind"] = []
        
        for time_hsk in data_hsk['tmhr'][mask]:            
            vars()[f"ssfr_logics_select_{i}_ind"].append(np.argmin(np.abs(t_ssfr - time_hsk)))
            vars()[f"hsr1_logics_select_{i}_ind"].append(np.argmin(np.abs(t_hsr1 - time_hsk)))
            
            
        vars()[f"ssfr_logics_select_{i}_ind"] = np.array(vars()[f"ssfr_logics_select_{i}_ind"])
        vars()[f"hsr1_logics_select_{i}_ind"] = np.array(vars()[f"hsr1_logics_select_{i}_ind"])
        
        vars()["cld_leg_%d" % i] = {}
        vars()["cld_leg_%d" % i]['time'] = data_hsk['tmhr'][mask]
        vars()["cld_leg_%d" % i]['lon'] = data_hsk['lon'][mask]
        vars()["cld_leg_%d" % i]['lat'] = data_hsk['lat'][mask]
        vars()["cld_leg_%d" % i]['sza'] = data_hsk['sza'][mask]
        vars()["cld_leg_%d" % i]['saa'] = data_hsk['saa'][mask]
        vars()["cld_leg_%d" % i]['ang_head'] = data_hsk['ang_hed'][mask]
        vars()["cld_leg_%d" % i]['ang_pit'] = data_hsk['ang_pit'][mask]
        vars()["cld_leg_%d" % i]['ang_rol'] = data_hsk['ang_rol'][mask]
        vars()["cld_leg_%d" % i]['ssfr_nad'] = data_ssfr['f_up'][vars()[f'ssfr_logics_select_{i}_ind'], :]
        vars()["cld_leg_%d" % i]['ssfr_zen'] = data_ssfr['f_dn'][vars()[f'ssfr_logics_select_{i}_ind'], :]
        # vars()["cld_leg_%d" % i]['ssfr_sza'] = data_ssfr['sza'][vars()[f'ssfr_logics_select_{i}_ind']]
        vars()["cld_leg_%d" % i]['hsr1_total'] = data_hsr1['f_dn_tot'][vars()[f'hsr1_logics_select_{i}_ind']]
        vars()["cld_leg_%d" % i]['hsr1_dif'] = data_hsr1['f_dn_dif'][vars()[f'hsr1_logics_select_{i}_ind']]
        vars()["cld_leg_%d" % i]['p3_alt'] = data_hsk['alt'][mask]/1000 # m to km
        
        if i == 0:
            for key in vars()["cld_leg_%d" % i].keys():
                vars()["cld_leg_all"][key] = vars()["cld_leg_%d" % i][key]
        else:
            for key in vars()["cld_leg_%d" % i].keys():
                vars()["cld_leg_all"][key] = np.concatenate((vars()["cld_leg_all"][key], vars()["cld_leg_%d" % i][key]), axis=0)
        
        
        # mask = np.abs(vars()["cld_leg_%d" % i]['ang_rol']+.0) > 0.8
        # vars()["cld_leg_%d" % i]['ssfr_zen'][mask, :] = np.nan
        # vars()["cld_leg_%d" % i]['ssfr_nad'][mask, :] = np.nan
        
        
        # plot alt-time, lwc_-time, hsr1-time, ssfr-time
    plt.close('all')
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    
    wvl_ind_ssfr_zen = np.argmin(np.abs(data_ssfr['wvl_dn'] - wvl_plot))
    print("wvl_ind_ssfr_zen:", wvl_ind_ssfr_zen)
    wvl_ind_ssfr_nad = np.argmin(np.abs(data_ssfr['wvl_up'] - wvl_plot)) 

    time_series_all = vars()["cld_leg_all"]['time']
    
    
    ax1.plot(time_series_all, vars()["cld_leg_all"]['ssfr_zen'][:, wvl_ind_ssfr_zen], 'o-', color='b', lw=1.0, markersize=5.0, label='SSFR zen flux')
    ax1.set_ylabel('Downward Flux (W m$^{-2} nm^{-1}$)')
    
    ax1_1 = ax1.twinx()
    sza_corrected_flux = vars()["cld_leg_all"]['ssfr_zen'][:, wvl_ind_ssfr_zen] / np.cos(np.deg2rad(vars()["cld_leg_all"]['sza']))
    ax1_1.plot(time_series_all, sza_corrected_flux, 'o-', color='darkred', lw=1.0, markersize=2.0, label='SZA corrected SSFR zen flux')
    ax1_1.set_ylabel('Flux normalized by SZA (W m$^{-2} nm^{-1}$)')

    
    
    ax2.plot(time_series_all, vars()["cld_leg_all"]['p3_alt'], 'o-', color='darkcyan', lw=1.0, markersize=5.0)
    ax2.set_ylabel('Altitude (km)')
    # ax2_2 = ax2.twinx()
    # ax2_2.plot(time_series_all, vars()["cld_leg_all"]['ang_rol'], 'o-', color='orange', lw=1.0, markersize=5.0, label='Roll Angle')
    # ax2_2.set_ylabel('Roll Angle (deg)')
    # ax2_2.set_ylim(-1.5, 1.5)
    
    ax3.plot(time_series_all, vars()["cld_leg_all"]['ang_head'], 'o-', color='g', lw=1.0, markersize=5.0, label='Heading')
    # ax3.plot(time_series_all, vars()["cld_leg_all"]['saa'], 'o-', color='r', lw=1.0, markersize=5.0, label='SAA')
    raa = np.mod(vars()["cld_leg_all"]['ang_head'] - vars()["cld_leg_all"]['saa'] - 180 + 75, 360)
    ax3.plot(time_series_all, raa, 'o-', color='orange', lw=1.0, markersize=5.0, label='Sun - relative Heading')
    ax3.set_ylabel('Angle (deg)')
    ax3.legend()
    ax3_3 = ax3.twinx()
    saa_minus_heading_adjust_deg = vars()["cld_leg_all"]['ang_head'] - vars()["cld_leg_all"]['saa'] + 75
    sin_sza_minus_heading_adjust_deg = np.sin(np.deg2rad(saa_minus_heading_adjust_deg))
    ax3_3.plot(time_series_all, sin_sza_minus_heading_adjust_deg, 'o-', color='purple', lw=1.0, markersize=5.0, label='sin(SZA - Heading - 75 deg)')
    ax3_3.set_ylabel('sin(Heading - SAA + 75$^o$)')
    
    cos_sza = np.cos(np.deg2rad(vars()["cld_leg_all"]['sza']))

    ax4.plot(time_series_all, cos_sza, 'o-', color='b', lw=1.0, markersize=5.0, label='cos(SZA)')
    ax4.set_ylabel('cos(SZA)')
    # ax4_4 = ax4.twinx()
    # ax4_4.plot(time_series_all, vars()["cld_leg_all"]['sza'], 'o-', color='r', lw=2.0, markersize=5.0, label='HSK SZA')
    # ax4_4.plot(time_series_all, vars()["cld_leg_all"]['ssfr_sza'], 'o-', color='pink', lw=1.0, markersize=2.5, label='SSFR SZA')
    # ax4_4.legend()
    # ax4_4.set_ylabel('SZA (deg)')
    
    
    ax4.set_xlabel('Time (UTC)')

        
    fig.suptitle(f'P3B LWC and HSR1/SSFR data for {date_s} - {time_series_all[0]:.2f} to {time_series_all[-1]:.2f}', fontsize=20)
    fig.tight_layout()
    fig.savefig(f'fig/{date_s}/P3B_SSFR_{date_s}_{time_series_all[0]:.2f}_{time_series_all[-1]:.2f}.png', bbox_inches='tight')
    plt.close(fig)
     

def marli_flt_trk_lrt_para(
        date: datetime.datetime = datetime.datetime(2024,6,5),
        extent = (-44,-58,83.4,84.1),
        sizes = (50,20,4),
        tmhr_ranges_select = ((15.36,15.6),(16.32,16.6)),
        fname_marli: str = "data/marli/…cdf",
        fname_kt19: str = "data/kt19/…ict",
        case_tag: str = "marli_test",
        config: Optional[FlightConfig] = None,
    ):
    log = logging.getLogger("marli")
    log.info("Starting processing for %s", date.strftime("%Y%m%d"))

    assert config, "FlightConfig required"
    date_s = date.strftime("%Y%m%d")

    # 1) Load all instrument & satellite metadata
    data_hsk  = load_h5(config.hsk(date_s))
    data_ssfr = load_h5(config.ssfr(date_s))
    data_hsr1 = load_h5(config.hsr1(date_s))
    data_logic= load_h5(config.logic(date_s))
    data_sat_coll = load_h5(config.sat_coll(date_s))
    head, data_kt19 = read_ict_kt19(str(config.kt19(fname_kt19)), encoding="utf-8",
                                    na_values=[-9999999,-777,-888])
    # MARLI netCDF
    with NcDataset(str(config.marli(fname_marli))) as ds:
        data_marli = {var: ds.variables[var][:] for var in ("time","Alt","H","T","LSR","WVMR")}

    # Satellite metadata
    sat_names, sat_tmhrs, sat_files = parse_sat(config.sat_coll(date_s))
    mid_times = np.array([np.mean(rng) for rng in tmhr_ranges_select])
    idx = closest_indices(sat_tmhrs, mid_times)
    
    
    # Solar spectrum
    f_solar = pd.read_csv('kurudz_ssfr.dat', delim_whitespace=True, comment='#', names=['wvl', 'flux'])
    wvl_solar = f_solar['wvl'].values
    flux_solar = f_solar['flux'].values/1000 # in W/m^2/nm
    flux_solar_interp = interp1d(wvl_solar, flux_solar, bounds_error=False, fill_value=0.0)


    # Build leg masks
    hsk_tm = np.array(data_hsk["tmhr"])
    leg_masks = [(hsk_tm>=lo)&(hsk_tm<=hi) for lo,hi in tmhr_ranges_select]

    # Loop legs: load raw NC, apply cloud logic, interpolate, plot
    for i, mask in enumerate(leg_masks):
        sat_nc = config.sat_nc(date_s, sat_files[idx[i]])
        # … load geolocation_data + geophysical_data …
        # … apply PCL fill, CGT table, thin cirrus conditioners …
        # … build interpolators (NearestNDInterpolator) …
        # … call two plotting helpers: one for map overlay, one for MARLI panels …
        
        
            
        # read the cloud observation information
        print('Reading cloud observation information from %s ...' % sat_nc)
        f = Dataset(sat_nc, 'r')
        lon_s = f.groups['geolocation_data'].variables['longitude'][...].data
        lat_s = f.groups['geolocation_data'].variables['latitude'][...].data

        cot_s = f.groups['geophysical_data'].variables['Cloud_Optical_Thickness'][...].data
        cot_pcl_s = f.groups['geophysical_data'].variables['Cloud_Optical_Thickness_PCL'][...].data

        cer_s = f.groups['geophysical_data'].variables['Cloud_Effective_Radius'][...].data
        cer_pcl_s = f.groups['geophysical_data'].variables['Cloud_Effective_Radius_PCL'][...].data
        
        cwp_s = np.float_(f.groups['geophysical_data'].variables['Cloud_Water_Path'][...].data)
        cwp_pcl_s = np.float_(f.groups['geophysical_data'].variables['Cloud_Water_Path_PCL'][...].data)
        
        cwp_s /= 1000 # convert g/m^2 to kg/m^2
        cwp_pcl_s /= 1000 # convert g/m^2 to kg/m

        cth_s = np.float_(f.groups['geophysical_data'].variables['Cloud_Top_Height'][...].data)
        
        cth_s /= 1000 # m to km
        
        ctt_s = np.float_(f.groups['geophysical_data'].variables['Cloud_Top_Temperature'][...].data)

        ctp_s = np.float_(f.groups['geophysical_data'].variables['Cloud_Phase_Optical_Properties'][...].data)

        scan_utc_s = f.groups['scan_line_attributes'].variables['scan_start_time'][...].data
        jday_s0 = np.array([er3t.util.dtime_to_jday(datetime.datetime(1993, 1, 1, 0, 0, 0) + datetime.timedelta(seconds=scan_utc_s[i])) for i in range(scan_utc_s.size)])
        f.close()

        logic_pcl = (cot_s<0.0) & (cot_pcl_s>0.0) & (cer_s<0.0) & (cer_pcl_s>0.0)
        cot_s[logic_pcl] = cot_pcl_s[logic_pcl]
        cer_s[logic_pcl] = cer_pcl_s[logic_pcl]
        cwp_s[logic_pcl] = cwp_pcl_s[logic_pcl]
        cwp_s[cwp_s<0.0] = np.nan

        logic_0 = (ctp_s==0.0)&((cot_s<0.0)|(cer_s<0.0))
        logic_1 = (ctp_s==1.0)&((cot_s<0.0)|(cer_s<0.0))
        logic_2 = (ctp_s==2.0)&((cot_s<0.0)|(cer_s<0.0))
        logic_3 = (ctp_s==3.0)&((cot_s<0.0)|(cer_s<0.0))
        logic_4 = (ctp_s==4.0)&((cot_s<0.0)|(cer_s<0.0))

        cot_s[logic_0] = np.nan
        cer_s[logic_0] = np.nan

        cot_s[logic_1] = 0.0
        cer_s[logic_1] = 1.0

        cot_s[logic_2] = -2.0
        cer_s[logic_2] = 1.0

        cot_s[logic_3] = -3.0
        cer_s[logic_3] = 1.0

        cot_s[logic_4] = -4.0
        cer_s[logic_4] = 1.0

        cot_s[cot_s<-10.0] = np.nan
        cer_s[cot_s<-10.0] = np.nan
        
        cgt_s = np.zeros_like(cth_s)
        cgt_s[...] = np.nan
        
        # read CGT_Data.csv
        cgt_table = pd.read_csv('CGT_Data.csv')
        for j in range(len(cgt_table)):
            pixel_select_cwp_low = np.logical_and(np.logical_and(cth_s > cgt_table['cth_low'][j], cth_s <= cgt_table['cth_high'][j]),
                                                cwp_s < cgt_table['cwp_threshold'][j]/1000)
            pixel_select_cwp_high = np.logical_and(np.logical_and(cth_s > cgt_table['cth_low'][j], cth_s <= cgt_table['cth_high'][j]),
                                                cwp_s >= cgt_table['cwp_threshold'][j]/1000)
            cgt_s[pixel_select_cwp_low] = cgt_table['a'][j]*cwp_s[pixel_select_cwp_low] + cgt_table['b'][j]
            cgt_s[pixel_select_cwp_high] = cgt_table['a_extra'][j]*cwp_s[pixel_select_cwp_high] + cgt_table['b_extra'][j]
            
        
        ext = [0.13, 0.25, 0.39, 0.55, 0.67] # in km-1
        ctt_thresh = [0, 200, 220, 240, 260, 400]
        # thin cirrus cloud
        for j in range(len(ctt_thresh)-1):
            pixel_select_thin = np.logical_and(cot_s>0, np.logical_and(cot_s < 1., np.logical_and(ctt_s > ctt_thresh[j], ctt_s <= ctt_thresh[j+1])))
            cgt_s[pixel_select_thin] = cot_s[pixel_select_thin]/ext[j]
            

        # sys.exit()
        cgt_s[cgt_s>cth_s] = cth_s[cgt_s>cth_s]
        cbh_s = cth_s - cgt_s
        
        cot_interp = NearestNDInterpolator(list(zip(lon_s.flatten(), lat_s.flatten())), cot_s.flatten())
        cer_interp = NearestNDInterpolator(list(zip(lon_s.flatten(), lat_s.flatten())), cer_s.flatten())
        cwp_interp = NearestNDInterpolator(list(zip(lon_s.flatten(), lat_s.flatten())), cwp_s.flatten())
        cth_interp = NearestNDInterpolator(list(zip(lon_s.flatten(), lat_s.flatten())), cth_s.flatten())
        cgt_interp = NearestNDInterpolator(list(zip(lon_s.flatten(), lat_s.flatten())), cgt_s.flatten())
        cbh_interp = NearestNDInterpolator(list(zip(lon_s.flatten(), lat_s.flatten())), cbh_s.flatten())
        
        cot_s0 = cot_s.copy()
        cot_s0[...] = 255.0
        # cs = ax1.pcolormesh(lon_s, lat_s, cot_s0, cmap='gray', vmin=0.0, vmax=20.0, zorder=0, transform=ccrs.PlateCarree(), alpha=1.0)
        cot_s[cot_s<=0.0] = np.nan
        ctp_s[ctp_s==1.0] = np.nan
        cth_s[cth_s<=0.0] = np.nan
        
        proj0 = ccrs.Orthographic(
                central_longitude=((extent[0]+extent[1])/2.0),
                central_latitude=((extent[2]+extent[3])/2.0),
                )
        plt.close('all')
        fig = plt.figure(figsize=(18, 12))
        ax1 = fig.add_subplot(211, projection=proj0)
        ax2 = fig.add_subplot(212, projection=proj0)
        cs_ctp = ax1.pcolormesh(lon_s, lat_s,  ctp_s, cmap='viridis', vmin=0.0, vmax=5.0, zorder=0, transform=ccrs.PlateCarree(), alpha=0.5)
        cs_cot = ax1.pcolormesh(lon_s, lat_s,  cot_s, cmap='jet', vmin=0.0, vmax=10.0, zorder=0, transform=ccrs.PlateCarree(), alpha=0.5)

        ax1.plot(data_hsk['lon'], data_hsk['lat'], lw=2.5, color='k', transform=ccrs.PlateCarree(), zorder=1)

        colors1 = ['r', 'g', 'b', 'brown']
        color = colors1[i]

        text1 = (date + datetime.timedelta(hours=tmhr_ranges_select[i][0])).strftime('%H:%M:%S')
        text2 = (date + datetime.timedelta(hours=tmhr_ranges_select[i][1])).strftime('%H:%M:%S')
        ax1.scatter(data_hsk['lon'][mask], data_hsk['lat'][mask], color=color, s=sizes[i], lw=0.0, alpha=1.0, transform=ccrs.PlateCarree())
        ax1.text(data_hsk['lon'][mask][0], data_hsk['lat'][mask][0]+0.03, text1, color=color, fontsize=16, alpha=1.0, va='bottom', ha='center', transform=ccrs.PlateCarree())
        ax1.text(data_hsk['lon'][mask][-1], data_hsk['lat'][mask][-1]+0.03, text2, color=color, fontsize=16, alpha=1.0, va='bottom', ha='center', transform=ccrs.PlateCarree())
        # ax1.scatter(f_lon[3], f_lat[3], s=5, marker='^', color='orange')
        # ax1.axvline(-52.3248, color='b', lw=1.0, alpha=1.0, zorder=0)
        # ax1.axvline(-51.7540, color='g', lw=1.0, alpha=1.0, zorder=0)
        # ax1.axvline(-51.3029, color='r', lw=1.0, alpha=1.0, zorder=0)


        
        cbar1 = fig.colorbar(cs_cot, ax=ax1, orientation='vertical', pad=0.05, shrink=1)
        cbar1.set_label('Cloud Optical Thickness', fontsize=16)
        cbar1.ax.tick_params(labelsize=14)
        
        
        
        # ax2 plot the cloud top height
        cs_cth = ax2.pcolormesh(lon_s, lat_s, cth_s, cmap='jet', vmin=0.0, vmax=8.0, zorder=0, transform=ccrs.PlateCarree(), alpha=0.5)
        ax2.plot(data_hsk['lon'], data_hsk['lat'], lw=2.5, color='k', transform=ccrs.PlateCarree(), zorder=1)


        text1 = (date + datetime.timedelta(hours=tmhr_ranges_select[i][0])).strftime('%H:%M:%S')
        text2 = (date + datetime.timedelta(hours=tmhr_ranges_select[i][1])).strftime('%H:%M:%S')
        ax2.scatter(data_hsk['lon'][mask], data_hsk['lat'][mask], color=color, s=sizes[i], lw=0.0, alpha=1.0, transform=ccrs.PlateCarree())
        ax2.text(data_hsk['lon'][mask][0], data_hsk['lat'][mask][0]+0.03, text1, color=color, fontsize=16, alpha=1.0, va='bottom', ha='center', transform=ccrs.PlateCarree())
        ax2.text(data_hsk['lon'][mask][-1], data_hsk['lat'][mask][-1]+0.03, text2, color=color, fontsize=16, alpha=1.0, va='bottom', ha='center', transform=ccrs.PlateCarree())
        # ax1.scatter(f_lon[3], f_lat[3], s=5, marker='^', color='orange')
        # ax1.axvline(-52.3248, color='b', lw=1.0, alpha=1.0, zorder=0)
        # ax1.axvline(-51.7540, color='g', lw=1.0, alpha=1.0, zorder=0)
        # ax1.axvline(-51.3029, color='r', lw=1.0, alpha=1.0, zorder=0)

        
        
        cbar2 = fig.colorbar(cs_cth, ax=ax2, orientation='vertical', pad=0.05, shrink=1, extend='max')
        cbar2.set_label('Cloud Top Height (km)', fontsize=16)
        cbar2.ax.tick_params(labelsize=14)
        
        
        for ax, label in zip([ax1, ax2], ['Cloud Optical Thickness', 'Cloud Top Height']):
            ax.coastlines(resolution='10m', color='gray', lw=0.5)
            g_lines = ax.gridlines(lw=0.5, color='gray', draw_labels=True, ls='-')
            g_lines.xlocator = FixedLocator(np.arange(-180, 181, 5.0))
            g_lines.ylocator = FixedLocator(np.arange(-90.0, 89.9, 0.2))
            g_lines.top_labels = False
            g_lines.right_labels = False
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.set_title(label, fontsize=16)
        
        
        sat_select_text = os.path.basename(sat_nc).replace('.nc', '').replace('CLDPROP_L2_', '')
        sat_ = sat_select_text.split('.')[0].replace('_', ' ')
        sat_utc = sat_select_text.split('.')[2]
        title_text = f'{sat_} {sat_utc}\n' + \
            f'Flight track {text1} - {text2} UTC'
        fig.suptitle(title_text, fontsize=20, y=0.95)
        

        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig('fig/%s/%s_%s_sat_%d_leg_%d.png' % (date_s, date_s, case_tag, idx[i], i), bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/# 
        
        # print("Marli time:", data_marli['tmhr'].min(), data_marli['tmhr'].max())

        # plot vertical marli data
        #/--------------------------------------------------------------/\#
        plt.close('all')
        
        data_marli_time = np.array(data_marli['time'])
        data_ssfr_time = np.array(data_ssfr['time'])/3600  # convert to hours
        data_hsr1_time = np.array(data_hsr1['time'])/3600  # convert to hours
        marli_select = []
        ssfr_select = []
        hsr1_select = []
        for time_hsk in data_hsk['tmhr'][mask]:
            marli_select.append(np.argmin(np.abs(data_marli_time - time_hsk)))
            ssfr_select.append(np.argmin(np.abs(data_ssfr_time - time_hsk)))
            hsr1_select.append(np.argmin(np.abs(data_hsr1_time - time_hsk)))
        
        marli_select = np.array(marli_select)
        ssfr_select = np.array(ssfr_select)
        hsr1_select = np.array(hsr1_select)
        
        marli_time = data_marli['time'][marli_select]
        marli_alt = data_marli['Alt'][marli_select]
        marli_H = data_marli['H']
        marli_T = np.array(data_marli['T'])[marli_select, :]
        marli_T[marli_T == 9999] = np.nan  # remove unrealistic temperatures
        marli_T[marli_T > 100] = np.nan  # remove unrealistic temperatures
        marli_T[marli_T < -100] = np.nan  # remove unrealistic temperatures
        marli_LSR = np.array(data_marli['LSR'])[marli_select, :]
        marli_LSR[marli_LSR == 9999] = np.nan  # remove unrealistic liquid water content
        marli_WVMR = np.array(data_marli['WVMR'])[marli_select, :]
        marli_WVMR[marli_WVMR == 9999] = np.nan  # remove unrealistic water vapor mixing ratios
        
        marli_time_repeat = np.repeat(marli_time[:, np.newaxis], marli_H.size, axis=1)
        marli_H_repeat = np.repeat(marli_H[np.newaxis, :], marli_time.size, axis=0)
        
        print("marli_H shape:", marli_H.shape)
        print("marli_time_repeat shape:", marli_time_repeat.shape)
        print("marli_H_repeat shape:", marli_H_repeat.shape)
        print("marli_T shape:", marli_T.shape)
        print("marli_LSR shape:", marli_LSR.shape)
        print("marli_WVMR shape:", marli_WVMR.shape)
        
        sza_hsk = data_hsk['sza'][mask]
        ssfr_zen_flux = data_ssfr['f_dn'][ssfr_select, :]
        ssfr_nad_flux = data_ssfr['f_up'][ssfr_select, :]
        ssfr_zen_toa = flux_solar_interp(data_ssfr['wvl_dn']) * np.cos(np.deg2rad(sza_hsk))[:, np.newaxis]  # W/m^2/nm
        ssfr_zen_wvl = data_ssfr['wvl_dn']
        ssfr_nad_wvl = data_ssfr['wvl_up']
        
        ssfr_nad_flux_interp = ssfr_zen_flux.copy()
        for j in range(ssfr_nad_flux.shape[0]):
            f_nad_flux_interp = interp1d(ssfr_nad_wvl, ssfr_nad_flux[j, :], axis=0, bounds_error=False, fill_value=np.nan)
            ssfr_nad_flux_interp[j, :] = f_nad_flux_interp(ssfr_zen_wvl)
        
        zen_795_left = np.argmin(np.abs(ssfr_zen_wvl - (795.0-4)))
        zen_795_right = np.argmin(np.abs(ssfr_zen_wvl - (795.0+4)))
        zen_1050_left = np.argmin(np.abs(ssfr_zen_wvl - (1050.0-4)))
        zen_1050_right = np.argmin(np.abs(ssfr_zen_wvl - (1050.0+4)))
        
        toa_795_avg = np.nanmean(ssfr_zen_toa[zen_795_left:zen_795_right+1])
        toa_1050_avg = np.nanmean(ssfr_zen_toa[zen_1050_left:zen_1050_right+1])
        
        zen_795_avg = np.nanmean(ssfr_zen_flux[zen_795_left:zen_795_right+1])
        zen_1050_avg = np.nanmean(ssfr_zen_flux[zen_1050_left:zen_1050_right+1])
        
        nad_795_avg = np.nanmean(ssfr_nad_flux_interp[zen_795_left:zen_795_right+1])
        nad_1050_avg = np.nanmean(ssfr_nad_flux_interp[zen_1050_left:zen_1050_right+1])
        
        zen_scaling = (zen_795_avg/zen_1050_avg) / (toa_795_avg/toa_1050_avg)
        nad_scaling = (nad_795_avg/nad_1050_avg) / (toa_795_avg/toa_1050_avg)
        
        zen_950_ind = np.argmin(np.abs(ssfr_zen_wvl - 950.0))
        
        ssfr_zen_flux[:, :zen_950_ind+1] /= zen_scaling
        ssfr_nad_flux_interp[:, :zen_950_ind+1] /= nad_scaling
        
        ssfr_nad_zen_ratio = ssfr_nad_flux_interp / ssfr_zen_flux
        
        wvl_550_ind = np.argmin(np.abs(ssfr_zen_wvl - 550.0))
        wvl_860_ind = np.argmin(np.abs(ssfr_zen_wvl - 860.0))
        wvl_1640_ind = np.argmin(np.abs(ssfr_zen_wvl - 1640.0))
        
        hsr1_wvl = data_hsr1['wvl_dn_tot']
        hsr1_total_flux = data_hsr1['f_dn_tot'][hsr1_select]
        hsr1_dif_flux = data_hsr1['f_dn_dif'][hsr1_select]
        hsr1_dif_ratio = hsr1_dif_flux / hsr1_total_flux
        hsr1_wvl_790_ind = np.argmin(np.abs(hsr1_wvl - 790.0))
        
        fig = plt.figure(figsize=(18, 12))
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        
        cc1 = ax1.contourf(marli_time_repeat, marli_H_repeat, marli_T, cmap='coolwarm', extend='both')
        ax1.set_xlabel('Time (UTC)', fontsize=16)
        ax1.set_ylabel('Height (km)', fontsize=16)
        cbar1 = fig.colorbar(cc1, ax=ax1, orientation='vertical', pad=0.05, shrink=1)
        cbar1.set_label('Temperature ($^o$C)', fontsize=16)
        cbar1.ax.tick_params(labelsize=14)
        ax1.set_title('MARLI Temperature', fontsize=20, y=1.01, color='k')
        
        cc2 = ax2.contourf(marli_time_repeat, marli_H_repeat, marli_LSR, cmap='jet', extend='both')
        ax2.set_xlabel('Time (UTC)', fontsize=16)
        ax2.set_ylabel('Height (km)', fontsize=16)
        cbar2 = fig.colorbar(cc2, ax=ax2, orientation='vertical', pad=0.05, shrink=1)
        cbar2.set_label('lidar scattering ratio', fontsize= 16)
        ax2.set_title('MARLI Lidar Scattering Ratio', fontsize=20, y=1.01, color='k')
        
        cc3 = ax3.contourf(marli_time_repeat, marli_H_repeat, marli_WVMR, cmap='Blues', extend='both')
        ax3.set_xlabel('Time (UTC)', fontsize=16)
        ax3.set_ylabel('Height (km)', fontsize=16)
        cbar3 = fig.colorbar(cc3, ax=ax3, orientation='vertical', pad=0.05, shrink=1)
        cbar3.set_label('Water Vapor Mixing Ratio (g/kg)', fontsize=16)
        ax3.set_title('MARLI Water Vapor Mixing Ratio', fontsize=20, y=1.01, color='k')
        
        ymax = np.round(np.nanmax(marli_alt)*1.1, 1)
        for ax in [ax1, ax2, ax3]:
            ax.plot(marli_time, marli_alt, color='k', lw=2.0)
            ax.set_ylim(0, ymax)
        
        color_list = ssfr_zen_flux[:, wvl_550_ind]/ np.nanmax(ssfr_zen_flux[:, wvl_550_ind])
        color_list[np.isnan(color_list)] = 0
        
        cc4 = ax4.scatter(data_ssfr_time[ssfr_select], ssfr_nad_zen_ratio[:, wvl_550_ind], c=color_list, s=0.5)
        ax4.plot(data_ssfr_time[ssfr_select], ssfr_nad_zen_ratio[:, wvl_550_ind], label='SSFR 550 nm', lw=2.0)
        ax4.plot(data_ssfr_time[ssfr_select], ssfr_nad_zen_ratio[:, wvl_860_ind], label='SSFR 860 nm', lw=2.0)
        ax4.plot(data_ssfr_time[ssfr_select], ssfr_nad_zen_ratio[:, wvl_1640_ind], label='SSFR 1640 nm', lw=2.0)
        ax4.plot(data_hsr1_time[hsr1_select], hsr1_dif_ratio[:, hsr1_wvl_790_ind], label='HSR1 790 nm diffuse', lw=2.0)
        ax4.legend(fontsize=16)
        ax4.set_xlabel('Time (UTC)', fontsize=16)
        ax4.set_ylabel('Nadir to Zenith Ratio', fontsize=16)
        ax4.set_title('SSFR upward/downward flux ratio\nHSR1 Diffuse ratio', fontsize=20, y=1.01, color='k')
        
        cbar4 = fig.colorbar(cc4, ax=ax4, orientation='vertical', pad=0.05, shrink=1)
        
        for ax in [ax1, ax2, ax3, ax4]:
            ax.tick_params(labelsize=14)
            time_interval = len(data_ssfr_time[ssfr_select])//5
            ax.set_xticks(data_ssfr_time[ssfr_select][::time_interval])
            ax.set_xticklabels([datetime.datetime.utcfromtimestamp(t*60*60).strftime('%H:%M') for t in data_ssfr_time[ssfr_select][::time_interval]])
        
        sat_select_text = os.path.basename(sat_nc).replace('.nc', '').replace('CLDPROP_L2_', '')
        sat_ = sat_select_text.split('.')[0].replace('_', ' ')
        sat_utc = sat_select_text.split('.')[2]
        title_text = f'{sat_} {sat_utc}\n' + \
            f'Flight track {text1} - {text2} UTC'
            
        fig.suptitle(title_text, fontsize=24, color='k')
        
        
        fig.tight_layout()
        fig.savefig('fig/%s/%s_%s_marli_ssfr_%d_leg_%d.png' % (date_s, date_s, case_tag, idx[i], i), bbox_inches='tight')
        
        cot_ssfr = cot_interp(data_ssfr['gps_lon'][ssfr_select], data_ssfr['gps_lat'][ssfr_select])
        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(111)
        ax1_1 = ax1.twinx()
        l1 = ax1.plot(data_ssfr_time[ssfr_select], ssfr_nad_zen_ratio[:, wvl_860_ind], label='SSFR 860 nm', color='blue', lw=2.0)
        l2 = ax1.plot(data_hsr1_time[hsr1_select], hsr1_dif_ratio[:, hsr1_wvl_790_ind], label='HSR1 790 nm diffuse', color='green', lw=2.0)
        l3 = ax1_1.plot(data_ssfr_time[ssfr_select], cot_ssfr, label='Satellite COT', color='orange', lw=2.0)
        ax1.set_xlabel('Time (UTC)', fontsize=16)
        ax1.set_ylabel('Ratio', fontsize=16)
        ax1_1.set_ylabel('Satellite COT', fontsize=16, color='orange')
        line_lebels = [l.get_label() for l in l1+l2+l3]
        ax1.legend(line_lebels, fontsize=16, loc='best')
        sat_select_text = os.path.basename(sat_nc).replace('.nc', '').replace('CLDPROP_L2_', '')
        sat_ = sat_select_text.split('.')[0].replace('_', ' ')
        sat_utc = sat_select_text.split('.')[2]
        title_text = f'{sat_} {sat_utc}\n' + \
            f'Flight track {text1} - {text2} UTC'
            
        fig.suptitle(title_text, fontsize=24, color='k')
        
        fig.tight_layout()
        fig.savefig('fig/%s/%s_%s_ssfr_hsr1_cot_%d_leg_%d.png' % (date_s, date_s, case_tag, idx[i], i), bbox_inches='tight')


    log.info("Finished.")                
    
def dropsonde_time_loc_list(dir_dropsonde='data/dropsonde'):
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

def gases_insitu(date, gas_dir, 
                         config: Optional[FlightConfig] = None,
                         plot=False,
                         time_select_range=None,):
    
    log = logging.getLogger("Gases Profile")
    log.info("Starting processing for %s", date.strftime("%Y%m%d"))

    assert config, "FlightConfig required"
    date_s = date.strftime("%Y%m%d")

    # 1) Load hsk and gases data
    data_hsk  = load_h5(config.hsk(date_s))
    h2o_filename = f"{gas_dir}/ARCSIX-DLH-H2O_P3B_{date_s}_R0.ict"
    ch4_filename = f"{gas_dir}/ARCSIX-TraceGas-CH4_P3B_{date_s}_R0.ict"
    co2_filename = f"{gas_dir}/ARCSIX-TraceGas-CO2_P3B_{date_s}_R0.ict"
    co_filename = f"{gas_dir}/ARCSIX-TraceGas-CO_P3B_{date_s}_R0.ict"
    o3_filename = f"{gas_dir}/ARCSIX-TraceGas-O3_P3B_{date_s}_R0.ict"
    head, data_dlh_h2o = read_ict_dlh_h2o(h2o_filename, encoding="utf-8", na_values=[-9999,-777,-888])
    head, data_ch4 = read_ict_ch4(ch4_filename, encoding="utf-8", na_values=[-9999,-777,-888])
    head, data_co2 = read_ict_co2(co2_filename, encoding="utf-8", na_values=[-9999,-777,-888])
    head, data_co = read_ict_co(co_filename, encoding="utf-8", na_values=[-9999,-777,-888])
    head, data_o3 = read_ict_o3(o3_filename, encoding="utf-8", na_values=[-9999,-777,-888])
    
    dropsonde_file_list, dropsonde_date_list, dropsonde_tmhr_list, dropsonde_lon_list, dropsonde_lat_list = dropsonde_time_loc_list()
    date_select = dropsonde_date_list == date.date()
    if np.sum(date_select) == 0:
        raise ValueError(f"No dropsonde data found for date {date.strftime('%Y-%m-%d')}")
    data_dropsonde_alt = []
    data_dropsonde_h2o = []
    for dropsonde_file in np.array(dropsonde_file_list)[date_select]:
        head, data_dropsonde_tmp = read_ict_dropsonde(dropsonde_file, encoding='utf-8', na_values=[-9999999, -777, -888])
        data_dropsonde_alt.extend(data_dropsonde_tmp['alt']/1000)  # convert to km
        data_dropsonde_h2o.extend(data_dropsonde_tmp['h2o_mr'])  # in g/kg
    data_dropsonde_alt = np.array(data_dropsonde_alt)
    data_dropsonde_h2o = np.array(data_dropsonde_h2o)
    data_dropsonde_h2o = data_dropsonde_h2o / 18.01528 / 1000 * 28.97 * 1e6  # convert to ppmv
    
    
    # 2) Get the altitude from hsk
    hsk_time = np.array(data_hsk['tmhr'])
    hsk_alt = np.array(data_hsk['alt'])/1000  # convert to km
    hsk_alt_interp = interp1d(hsk_time, hsk_alt, bounds_error=False, fill_value=np.nan)
    
    # 3) Interpolate altitude to gases data time
    dlh_h2o_time = np.array(data_dlh_h2o['tmhr'])
    ch4_time = np.array(data_ch4['tmhr'])
    co2_time = np.array(data_co2['tmhr'])
    co_time = np.array(data_co['tmhr'])
    o3_time = np.array(data_o3['tmhr'])
    
    dlh_h2o_alt = hsk_alt_interp(dlh_h2o_time)
    ch4_alt = hsk_alt_interp(ch4_time)
    co2_alt = hsk_alt_interp(co2_time)
    co_alt = hsk_alt_interp(co_time)
    o3_alt = hsk_alt_interp(o3_time)
    
    # 4) calculate the gases profiles based on altitude bins
    alt_bins = np.arange(0, 10, 0.1)  # 0 to 10 km, 0.1 km interval
    alt_bin_centers = (alt_bins[:-1] + alt_bins[1:]) / 2
    h2o_profile = np.full(alt_bin_centers.shape, np.nan)
    ch4_profile = np.full(alt_bin_centers.shape, np.nan)
    co2_profile = np.full(alt_bin_centers.shape, np.nan)
    co_profile = np.full(alt_bin_centers.shape, np.nan)
    o3_profile = np.full(alt_bin_centers.shape, np.nan)
    
    h2o_dropsonde_profile = np.full(alt_bin_centers.shape, np.nan)
    
    h2o_profile_unc = np.full(alt_bin_centers.shape, np.nan)
    ch4_profile_unc = np.full(alt_bin_centers.shape, np.nan)
    co2_profile_unc = np.full(alt_bin_centers.shape, np.nan)
    co_profile_unc = np.full(alt_bin_centers.shape, np.nan)
    o3_profile_unc = np.full(alt_bin_centers.shape, np.nan)
    
    h2o_dropsonde_profile_unc = np.full(alt_bin_centers.shape, np.nan)
    
    for i in range(len(alt_bin_centers)):
        alt_min = alt_bins[i]
        alt_max = alt_bins[i+1]
        
        # H2O
        mask_h2o = (dlh_h2o_alt >= alt_min) & (dlh_h2o_alt < alt_max) & (~np.isnan(data_dlh_h2o['h2o_vmr']))
        if np.any(mask_h2o):
            h2o_profile[i] = np.nanmean(data_dlh_h2o['h2o_vmr'][mask_h2o])
            h2o_profile_unc[i] = np.nanstd(data_dlh_h2o['h2o_vmr'][mask_h2o]) / np.sqrt(np.sum(mask_h2o))
            
        # Dropsonde H2O
        mask_h2o_dropsonde = (data_dropsonde_alt >= alt_min) & (data_dropsonde_alt < alt_max)
        if np.any(mask_h2o_dropsonde):
            h2o_dropsonde_profile[i] = np.nanmean(data_dropsonde_h2o[mask_h2o_dropsonde])
            h2o_dropsonde_profile_unc[i] = np.nanstd(data_dropsonde_h2o[mask_h2o_dropsonde]) / np.sqrt(np.sum(mask_h2o_dropsonde))
        
        # CH4
        mask_ch4 = (ch4_alt >= alt_min) & (ch4_alt < alt_max) & (~np.isnan(data_ch4['ch4']))
        if np.any(mask_ch4):
            ch4_profile[i] = np.nanmean(data_ch4['ch4'][mask_ch4])
            ch4_profile_unc[i] = np.nanstd(data_ch4['ch4'][mask_ch4]) / np.sqrt(np.sum(mask_ch4))
        
        # CO2
        mask_co2 = (co2_alt >= alt_min) & (co2_alt < alt_max) & (~np.isnan(data_co2['co2']))
        if np.any(mask_co2):
            co2_profile[i] = np.nanmean(data_co2['co2'][mask_co2])
            co2_profile_unc[i] = np.nanstd(data_co2['co2'][mask_co2]) / np.sqrt(np.sum(mask_co2))
        
        # CO
        mask_co = (co_alt >= alt_min) & (co_alt < alt_max) & (~np.isnan(data_co['co']))
        if np.any(mask_co):
            co_profile[i] = np.nanmean(data_co['co'][mask_co])
            co_profile_unc[i] = np.nanstd(data_co['co'][mask_co]) / np.sqrt(np.sum(mask_co))
        
        # O3
        mask_o3 = (o3_alt >= alt_min) & (o3_alt < alt_max) & (~np.isnan(data_o3['o3']))
        if np.any(mask_o3):
            o3_profile[i] = np.nanmean(data_o3['o3'][mask_o3])
            o3_profile_unc[i] = np.nanstd(data_o3['o3'][mask_o3]) / np.sqrt(np.sum(mask_o3))
    
    # 5) Save the profiles to a csv file
    df_profiles = pd.DataFrame({
        'Altitude_km': alt_bin_centers,
        'H2O_VMR_ppm': h2o_profile,
        'H2O_VMR_unc_ppm': h2o_profile_unc,
        'H2O_Dropsonde_VMR_ppm': h2o_dropsonde_profile,
        'H2O_Dropsonde_VMR_unc_ppm': h2o_dropsonde_profile_unc,
        'CH4_VMR_ppm': ch4_profile,
        'CH4_VMR_unc_ppm': ch4_profile_unc,
        'CO2_VMR_ppm': co2_profile,
        'CO2_VMR_unc_ppm': co2_profile_unc,
        'CO_VMR_ppb': co_profile,
        'CO_VMR_unc_ppb': co_profile_unc,
        'O3_VMR_ppb': o3_profile,
        'O3_VMR_unc_ppb': o3_profile_unc,
    })
    os.makedirs(f'data/zpt/{date_s}', exist_ok=True)
    df_profiles.to_csv(f'data/zpt/{date_s}/{date_s}_gases_profiles.csv', index=False)
    log.info(f"Saved gases profiles to data/zpt/{date_s}/{date_s}_gases_profiles.csv")
    
    
    if plot:
        # plot the gases profiles
        plt.close('all')
        fig, axs = plt.subplots(1, 5, figsize=(20, 6))
        ax1, ax2, ax3, ax4, ax5 = axs
        ax1.errorbar(h2o_profile, alt_bin_centers, xerr=h2o_profile_unc, fmt='b.-', alpha=0.7)
        ax1.plot(h2o_profile, alt_bin_centers, 'b.-', label='DLH in-situ')
        ax1.errorbar(h2o_dropsonde_profile, alt_bin_centers, xerr=h2o_dropsonde_profile_unc, fmt='k.--', alpha=0.7)
        ax1.plot(h2o_dropsonde_profile, alt_bin_centers, 'k.--', label='Dropsonde')
        ax1.legend(fontsize=12)
        ax1.set_xlabel('H$_2$O VMR (ppm)', fontsize=14)
        ax1.set_ylabel('Altitude (km)', fontsize=14)
        ax1.set_title('H$_2$O Profile', fontsize=16)
        
        ax2.errorbar(ch4_profile, alt_bin_centers, xerr=ch4_profile_unc, fmt='g.-', alpha=0.7)
        ax2.plot(ch4_profile, alt_bin_centers, 'g.-')
        ax2.set_xlabel('CH$_4$ VMR (ppb)', fontsize=14)
        ax2.set_title('CH$_4$ Profile', fontsize=16)
        ax2.yaxis.set_visible(False)
        
        ax3.errorbar(co2_profile, alt_bin_centers, xerr=co2_profile_unc, fmt='r.-', alpha=0.7)
        ax3.plot(co2_profile, alt_bin_centers, 'r.-')
        ax3.set_xlabel('CO$_2$ VMR (ppm)', fontsize=14)
        ax3.set_title('CO$_2$ Profile', fontsize=16)
        ax3.yaxis.set_visible(False)
        
        ax4.errorbar(co_profile, alt_bin_centers, xerr=co_profile_unc, fmt='m.-', alpha=0.7)
        ax4.plot(co_profile, alt_bin_centers, 'm.-')
        ax4.set_xlabel('CO VMR (ppb)', fontsize=14)
        ax4.set_title('CO Profile', fontsize=16)
        ax4.yaxis.set_visible(False)
        
        ax5.errorbar(o3_profile, alt_bin_centers, xerr=o3_profile_unc, fmt='c.-', alpha=0.7)        
        ax5.plot(o3_profile, alt_bin_centers, 'c.-')
        ax5.set_xlabel('O$_3$ VMR (ppb)', fontsize=14)
        ax5.set_title('O$_3$ Profile', fontsize=16)
        ax5.yaxis.set_visible(False)
        for ax in axs:
            ax.tick_params(labelsize=12)
            ax.grid(True)
            ax.set_ylim(0, 10)
        fig.tight_layout(pad=2.0)
        fig.suptitle(f'Gases Vertical Profiles {date_s}', fontsize=18, y=1.02)
        fig.savefig('data/zpt/%s/%s_gases_profiles.png' % (date_s, date_s), bbox_inches='tight')
        # plt.show()
        
        if time_select_range is not None:
            for time_start, time_end in time_select_range:
                time_mask_hsk = (data_hsk['tmhr']>=time_start) & (data_hsk['tmhr']<=time_end)
                time_mask_dlh_h2o = (data_dlh_h2o['tmhr']>=time_start) & (data_dlh_h2o['tmhr']<=time_end)
                time_mask_ch4 = (data_ch4['tmhr']>=time_start) & (data_ch4['tmhr']<=time_end)
                time_mask_co2 = (data_co2['tmhr']>=time_start) & (data_co2['tmhr']<=time_end)
                time_mask_co = (data_co['tmhr']>=time_start) & (data_co['tmhr']<=time_end)
                time_mask_o3 = (data_o3['tmhr']>=time_start) & (data_o3['tmhr']<=time_end)
                
                
                # plot time series
                plt.close('all')
                fig, axs = plt.subplots(2, 2, figsize=(14, 8))
                ax1 = axs[0, 0]
                ax2 = axs[0, 1]
                ax3 = axs[1, 0]
                ax4 = axs[1, 1]
                
                ax1.plot(dlh_h2o_time[time_mask_dlh_h2o], data_dlh_h2o['h2o_vmr'][time_mask_dlh_h2o], 'b.-', label='DLH H2O VMR')
                ax1.set_ylabel('H2O VMR (ppm)', fontsize=14)
                ax1_1 = ax1.twinx()
                ax1_1.plot(co2_time[time_mask_co2], data_co2['co2'][time_mask_co2], 'r.-', label='CO2 VMR')
                ax1_1.set_ylabel('CO2 VMR (ppm)', fontsize=14, color='r')
                ax1.set_xlabel('Time (hours)', fontsize=14)
                
                ax2.plot(ch4_time[time_mask_ch4], data_ch4['ch4'][time_mask_ch4], 'g.-', label='CH4 VMR')
                ax2.set_ylabel('CH4 VMR (ppb)', fontsize=14)
                ax2_1 = ax2.twinx()
                ax2_1.plot(co2_time[time_mask_co2], data_co2['co2'][time_mask_co2], 'r.-', label='CO2 VMR')
                ax2.set_xlabel('Time (hours)', fontsize=14)
                
                ax3.plot(co_time[time_mask_co], data_co['co'][time_mask_co], 'm.-', label='CO VMR')
                ax3.set_ylabel('CO VMR (ppb)', fontsize=14)
                ax3_1 = ax3.twinx()
                ax3_1.plot(co2_time[time_mask_co2], data_co2['co2'][time_mask_co2], 'r.-', label='CO2 VMR')
                ax3_1.set_ylabel('CO2 VMR (ppm)', fontsize=14, color='r')
                ax3.set_xlabel('Time (hours)', fontsize=14)
                
                ax4.plot(o3_time[time_mask_o3], data_o3['o3'][time_mask_o3], 'c.-', label='O3 VMR')
                ax4.set_ylabel('O3 VMR (ppb)', fontsize=14)
                ax4_1 = ax4.twinx()
                ax4_1.plot(co2_time[time_mask_co2], data_co2['co2'][time_mask_co2], 'r.-', label='CO2 VMR')
                ax4_1.set_ylabel('CO2 VMR (ppm)', fontsize=14, color='r')
                ax4.set_xlabel('Time (hours)', fontsize=14) 
                
                for ax in [ax1, ax2, ax3, ax4]:
                    ax.tick_params(labelsize=12)
                    # ax.grid(True)
                    ax.legend(fontsize=12)
                fig.suptitle(f'Gases Time Series {date_s} {time_start:.2f}-{time_end:.2f} UTC', fontsize=16)
                fig.tight_layout(pad=2.0)
                plt.show()
 


def flt_trk_atm_corr(date=datetime.datetime(2024, 5, 31),
                     extent=[-60, -80, 82.4, 84.6],
                     tmhr_ranges_select=[[14.10, 14.27]],
                     modis_07_file=['./data/sat-data/20240531/MOD07_L2.A2024152.1525.061.2024153011814.hdf'],
                     case_tag='default',
                     config: Optional[FlightConfig] = None,
                     levels=None,
                     simulation_interval=3,
                     clear_sky=True,
                     overwrite_atm=False,
                     overwrite_alb=False,
                     overwrite_lrt=True,
                     manual_cloud=False,
                     manual_cloud_cer=14.4,
                     manual_cloud_cwp=0.06013,
                     manual_cloud_cth=0.945,
                     manual_cloud_cbh=0.344,
                     manual_cloud_cot=6.26,
                     iter=0
                            ):

    # case specification
    #/----------------------------------------------------------------------------\#
    vname_x = 'lon'
    colors1 = ['r', 'g', 'b', 'brown']
    colors2 = ['hotpink', 'springgreen', 'dodgerblue', 'orange']
    #\----------------------------------------------------------------------------/#
    
    
    log = logging.getLogger("lrt")
    log.info("Starting processing for %s", date.strftime("%Y%m%d"))

    assert config, "FlightConfig required"
    date_s = date.strftime("%Y%m%d")

    # 1) Load all instrument & satellite metadata
    data_hsk  = load_h5(config.hsk(date_s))
    data_ssfr = load_h5(config.ssfr(date_s))
    data_hsr1 = load_h5(config.hsr1(date_s))
    
    # Solar spectrum
    f_solar = pd.read_csv('kurudz_ssfr.dat', delim_whitespace=True, comment='#', names=['wvl', 'flux'])
    wvl_solar = f_solar['wvl'].values
    flux_solar = f_solar['flux'].values/1000 # in W/m^2/nm
    flux_solar_interp = interp1d(wvl_solar, flux_solar, bounds_error=False, fill_value=0.0)


    # Build leg masks
    t_hsk = np.array(data_hsk["tmhr"])
    leg_masks = [(t_hsk>=lo)&(t_hsk<=hi) for lo,hi in tmhr_ranges_select]
    
    t_ssfr = data_ssfr['time']/3600.0  # convert to hours
    t_hsr1 = data_hsr1['time']/3600.0  # convert to hours
    

    
    # create atmospheric profile
    #/----------------------------------------------------------------------------\#
    dropsonde_file_list, dropsonde_date_list, dropsonde_tmhr_list, dropsonde_lon_list, dropsonde_lat_list = dropsonde_time_loc_list()
    
    date_select = dropsonde_date_list == date.date()
    if np.sum(date_select) == 0:
        raise ValueError(f"No dropsonde data found for date {date.strftime('%Y-%m-%d')}")
    dropsonde_tmhr_array = np.array(dropsonde_tmhr_list)[date_select]
    # find the closest dropsonde time to the flight mid times
    mid_tmhr = np.array([np.mean(rng) for rng in tmhr_ranges_select])
    dropsonde_idx = closest_indices(dropsonde_tmhr_array, mid_tmhr)
    dropsonde_file = np.array(dropsonde_file_list)[date_select][dropsonde_idx[0]]
    log.info(f"Using dropsonde file: {dropsonde_file}")
    head, data_dropsonde = read_ict_dropsonde(dropsonde_file, encoding='utf-8', na_values=[-9999999, -777, -888])
    # log.info(f"Dropsonde time: {data_dropsonde['time'][0]} UTsC")
    
    
    
    zpt_filedir = f'{_fdir_general_}/zpt/{date_s}'
    os.makedirs(zpt_filedir, exist_ok=True)
    if levels is None:
        levels = np.concatenate((np.arange(0, 2.1, 0.2), 
                                np.arange(2.5, 4.1, 0.5), 
                                np.arange(5.0, 10.1, 2.5),
                                np.array([15, 20, 30., 40., 50.])))
    if not os.path.exists(os.path.join(zpt_filedir, f'atm_profiles_{date_s}_{case_tag}.dat')) or overwrite_atm:
        lon = np.mean(extent[:2])
        lat = np.mean(extent[2:])
        boundary_from_center = 0.25 # degree
        mod_lon = np.array([lon-boundary_from_center, lon+boundary_from_center])
        mod_lat = np.array([lat-boundary_from_center, lat+boundary_from_center])
        mod_extent = [mod_lon[0], mod_lon[1], mod_lat[0], mod_lat[1]]
        
        zpt_filename = f'zpt_{date_s}_{case_tag}.h5'
        
        fname_atm = f'modis_dropsonde_atm_{date_s}_{case_tag}.pk'
        
        status, ws10m = er3t.pre.atm.create_modis_dropsonde_atm(o2mix=0.20935, output_dir=zpt_filedir, output=zpt_filename, 
                                                fname_mod07=modis_07_file, dropsonde_df=data_dropsonde,
                                                levels=levels,
                                                extent=mod_extent, new_h_edge=None,sfc_T_set=None, sfc_h_to_zero=True,)
        
        atm0      = er3t.pre.atm.modis_dropsonde_arcsix_atmmod(zpt_file=f'{zpt_filedir}/{zpt_filename}',
                            fname=fname_atm, 
                            fname_co2_clim=f'{_fdir_general_}/climatology/cams73_latest_co2_conc_surface_inst_2020.nc',
                            fname_ch4_clim=f'{_fdir_general_}/climatology/cams_ch4_202005-202008.nc',
                            fname_o3_clim=f'{_fdir_general_}/climatology/ozone_merra2_202405_202408.h5',
                            fname_insitu=f'data/zpt/{date_s}/{date_s}_gases_profiles.csv',
                            date=date, extent=mod_extent,
                            overwrite=True)
    
        # write out the atmospheric profile in ascii format
        with open(os.path.join(zpt_filedir, f'atm_profiles_{date_s}_{case_tag}.dat'), 'w') as f:
            header = ('# Adjusted MODIS 07 atmospheric profile\n'
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
        
        with open(f'{zpt_filedir}/ch4_profiles_{date_s}_{case_tag}.dat', 'w') as f:  
            header = ('# Adjusted MODIS 07 atmospheric profile for ch4 only\n'
                    '#      z(km)      ch4(cm-3)\n'
                    )
            lines = [
                    f'{atm0.lev["altitude"]["data"][i]:11.3f} {atm0.lev["ch4"]["data"][i]:12.6e}'
                    for i in range(len(atm0.lev['altitude']['data']))[::-1]
                    ]
            f.write(header + "\n".join(lines))
    # =================================================================================
 
    
    xx = np.linspace(-12, 12, 241)
    yy_gaussian_vis = gaussian(xx, 0, 3.82)
    yy_gaussian_nir = gaussian(xx, 0, 5.10)
    with open(os.path.join('.', 'vis_0.1nm_0710.dat'), 'w') as f_slit:
        header = ('# SSFR Silicon slit function\n'
                    '# wavelength (nm)      relative intensity\n'
                    )
        # Build all profile lines in one go.
        lines = [
                 f'{xx[i]:11.1f} '
                 f'{yy_gaussian_vis[i]:12.5e}'
                 for i in range(len(xx))
                ]
        f_slit.write(header + "\n".join(lines))
        
    xx_wvl_grid = np.arange(360, 1990.1, 5)
    with open(os.path.join('.', 'wvl_grid_test.dat'), 'w') as f_grid:
        # Build all profile lines in one go.
        lines = [
                 f'{xx_wvl_grid[i]:11.1f} '
                 for i in range(len(xx_wvl_grid))
                ]
        f_grid.write(header + "\n".join(lines))
    
    # write out the convolved solar flux
    #/----------------------------------------------------------------------------\#
    wvl_solar_vis = np.arange(300, 950.1, 1)
    wvl_solar_nir = np.arange(951, 2500.1, 1)
    wvl_solar_coarse = np.concatenate([wvl_solar_vis, wvl_solar_nir])
    effective_wvl = wvl_solar_coarse[np.logical_and(wvl_solar_coarse >= xx_wvl_grid.min(), wvl_solar_coarse <= xx_wvl_grid.max())]
    if 1:#not os.path.exists('kurudz_ssfr.dat') or overwrite_lrt:
        df_solor = pd.read_csv('kurudz_0.1nm.dat', sep='\s+', header=None)
        wvl_solar = np.array(df_solor.iloc[:, 0])
        flux_solar = np.array(df_solor.iloc[:, 1])#/1000 # convert mW/m^2/nm to W/m^2/nm
        mask = wvl_solar <= 2500

        wvl_solar = wvl_solar[mask]
        flux_solar = flux_solar[mask]
        flux_solar_convolved = flux_solar.copy()
        
        assert (xx[1]-xx[0]) - (wvl_solar[1]-wvl_solar[0]) <1e-3

        flux_solar_convolved_vis = convolve(flux_solar, yy_gaussian_vis, mode='same') / np.sum(yy_gaussian_vis)
        flux_solar_convolved_nir = convolve(flux_solar, yy_gaussian_nir, mode='same') / np.sum(yy_gaussian_nir)
        flux_solar_convolved[wvl_solar<=950] = flux_solar_convolved_vis[wvl_solar<=950]
        flux_solar_convolved[wvl_solar>950] = flux_solar_convolved_nir[wvl_solar>950]
        
        flux_solar_convolved_coarse = np.zeros_like(wvl_solar_coarse)
        for vis_wvl_i in range(len(wvl_solar_vis)):
            ind = wvl_solar == wvl_solar_vis[vis_wvl_i]
            flux_solar_convolved_coarse[vis_wvl_i] = flux_solar_convolved_vis[ind]
        for nir_wvl_i in range(len(wvl_solar_nir)):
            ind = wvl_solar == wvl_solar_nir[nir_wvl_i]
            flux_solar_convolved_coarse[nir_wvl_i+len(wvl_solar_vis)] = flux_solar_convolved_nir[ind]
        
        with open('kurudz_ssfr.dat', 'w') as f_solar:
            header = ('# SSFR version solar flux\n'
                    '# wavelength (nm)      flux (mW/m^2/nm)\n'
                    )
            # Build all profile lines in one go.
            lines = [
                    f'{wvl_solar_coarse[i]:11.1f} '
                    f'{flux_solar_convolved_coarse[i]:12.5e}'
                    for i in range(len(wvl_solar_coarse))
                    ]
            f_solar.write(header + "\n".join(lines))
        

    

    # read satellite granule
    #/----------------------------------------------------------------------------\#
    import platform
    
    fdir_cld_obs_info = f'{_fdir_general_}/flt_cld_obs_info'
    os.makedirs(fdir_cld_obs_info, exist_ok=True)
    fname_cld_obs_info = '%s/%s_cld_obs_info_%s_%s_%s_atm_corr.pkl' % (fdir_cld_obs_info, _mission_.lower(), _platform_.lower(), date_s, case_tag)
    if not os.path.exists(fname_cld_obs_info):      
        
        # Loop legs: load raw NC, apply cloud logic, interpolate, plot
        for i, mask in enumerate(leg_masks):
            
            # find index arrays in one go
            times_leg = t_hsk[mask]
            
            sel_ssfr, sel_hsr1 = (
                nearest_indices(t_hsk, mask, arr)
                for arr in (t_ssfr, t_hsr1)
            )
            

            # assemble a small dict for this leg
            leg = {
                "time":    times_leg,
                "alt":     data_hsk["alt"][mask] / 1000.0,
                "hsr1_tot": data_hsr1["f_dn_tot"][sel_hsr1],
                "hsr1_dif": data_hsr1["f_dn_dif"][sel_hsr1],
                # "ssfr_zen": data_ssfr["f_dn"][sel_ssfr],
                # "ssfr_nad": data_ssfr["f_up"][sel_ssfr],
                "hsr1_wvl": data_hsr1["wvl_dn_tot"],
                # "ssfr_zen_wvl": data_ssfr["wvl_dn"],
                # "ssfr_nad_wvl": data_ssfr["wvl_up"],
                "lon":     data_hsk["lon"][mask],
                "lat":     data_hsk["lat"][mask],
                "sza":     data_hsk["sza"][mask],
                "saa":     data_hsk["saa"][mask],
                "p3_alt":  data_hsk["alt"][mask] / 1000.0,
            }
            

                
            if clear_sky:
                leg.update({
                    "cot": np.full_like(leg['lon'], np.nan),
                    "cer": np.full_like(leg['lon'], np.nan),
                    "cwp": np.full_like(leg['lon'], np.nan),
                    "cth": np.full_like(leg['lon'], np.nan),
                    "cgt": np.full_like(leg['lon'], np.nan),
                    "cbh": np.full_like(leg['lon'], np.nan),
                })
            elif not clear_sky and manual_cloud:
                leg.update({
                    "cot": np.full_like(leg['lon'], manual_cloud_cot),
                    "cer": np.full_like(leg['lon'], manual_cloud_cer),
                    "cwp": np.full_like(leg['lon'], manual_cloud_cwp),
                    "cth": np.full_like(leg['lon'], manual_cloud_cth),
                    "cgt": np.full_like(leg['lon'], manual_cloud_cth-manual_cloud_cbh),
                    "cbh": np.full_like(leg['lon'], manual_cloud_cbh),
                })
            else:
                raise NotImplementedError("Automatic cloud retrieval not implemented yet")
            
            sza_hsk = data_hsk['sza'][mask]
            ssfr_zen_toa = flux_solar_interp(data_ssfr['wvl_dn']) * np.cos(np.deg2rad(sza_hsk))[:, np.newaxis]  # W/m^2/nm
            
            
            ssfr_zen_flux = data_ssfr['f_dn'][sel_ssfr, :]
            ssfr_nad_flux = data_ssfr['f_up'][sel_ssfr, :]
            ssfr_zen_toa = flux_solar_interp(data_ssfr['wvl_dn']) * np.cos(np.deg2rad(sza_hsk))[:, np.newaxis]  # W/m^2/nm
            ssfr_zen_wvl = data_ssfr['wvl_dn']
            ssfr_nad_wvl = data_ssfr['wvl_up']
            
            ssfr_nad_flux_interp = ssfr_zen_flux.copy()
            for j in range(ssfr_nad_flux.shape[0]):
                f_nad_flux_interp = interp1d(ssfr_nad_wvl, ssfr_nad_flux[j, :], axis=0, bounds_error=False, fill_value=np.nan)
                ssfr_nad_flux_interp[j, :] = f_nad_flux_interp(ssfr_zen_wvl)
            
            zen_795_left = np.argmin(np.abs(ssfr_zen_wvl - (795.0-4)))
            zen_795_right = np.argmin(np.abs(ssfr_zen_wvl - (795.0+4)))
            zen_1050_left = np.argmin(np.abs(ssfr_zen_wvl - (1050.0-4)))
            zen_1050_right = np.argmin(np.abs(ssfr_zen_wvl - (1050.0+4)))
            
            toa_795_avg = np.nanmean(ssfr_zen_toa[:, zen_795_left:zen_795_right+1], axis=1)
            toa_1050_avg = np.nanmean(ssfr_zen_toa[:, zen_1050_left:zen_1050_right+1], axis=1)
            
            zen_795_avg = np.nanmean(ssfr_zen_flux[:, zen_795_left:zen_795_right+1], axis=1)
            zen_1050_avg = np.nanmean(ssfr_zen_flux[:, zen_1050_left:zen_1050_right+1], axis=1)
            
            nad_795_avg = np.nanmean(ssfr_nad_flux_interp[:, zen_795_left:zen_795_right+1], axis=1)
            nad_1050_avg = np.nanmean(ssfr_nad_flux_interp[:, zen_1050_left:zen_1050_right+1], axis=1)
            
            zen_scaling = (zen_795_avg/zen_1050_avg) / (toa_795_avg/toa_1050_avg)
            nad_scaling = (nad_795_avg/nad_1050_avg) / (toa_795_avg/toa_1050_avg)
            
            
            # print(f'Leg {i}: zen_scaling = {zen_scaling:.3f}, nad_scaling = {nad_scaling:.3f}')
            
            zen_950_ind = np.argmin(np.abs(ssfr_zen_wvl - 950.0))
            

            ssfr_zen_flux[:, :zen_950_ind+1] /= zen_scaling[:, np.newaxis]
            ssfr_nad_flux_interp[:, :zen_950_ind+1] /= nad_scaling[:, np.newaxis]
            
            
            leg['ssfr_zen'] = ssfr_zen_flux
            leg['ssfr_nad'] = ssfr_nad_flux_interp
            leg['ssfr_zen_wvl'] = ssfr_zen_wvl
            leg['ssfr_nad_wvl'] = ssfr_zen_wvl
            leg['ssfr_toa'] = ssfr_zen_toa

            vars()["cld_leg_%d" % i] = leg
        
            # save the cloud observation information to a pickle file
            fname_pkl = '%s/%s_cld_obs_info_%s_%s_%s_%d_atm_corr.pkl' % (fdir_cld_obs_info, _mission_.lower(), _platform_.lower(), date_s, case_tag, i)
            with open(fname_pkl, 'wb') as f:
                pickle.dump(vars()["cld_leg_%d" % i], f, protocol=pickle.HIGHEST_PROTOCOL)
        
            # sys.exit()
    else:
        print('Loading cloud observation information from %s ...' % fname_cld_obs_info)
        for i in range(len(tmhr_ranges_select)):
            fname_pkl = '%s/%s_cld_obs_info_%s_%s_%s_%d_atm_corr.pkl' % (fdir_cld_obs_info, _mission_.lower(), _platform_.lower(), date_s, case_tag, i)
            with open(fname_pkl, 'rb') as f:
                vars()[f"cld_leg_{i}"] = pickle.load(f)   
    
    solver = 'lrt'
    
    for ileg, _ in enumerate(leg_masks):
        
        cld_leg = vars()[f'cld_leg_{ileg}'] 
        times_leg = cld_leg['time']
        time_start, time_end = times_leg[0], times_leg[-1]
        alt_avg = np.nanmean(cld_leg['alt'])
        
        if clear_sky:
            fname_h5 = '%s/%s-%s-%s-%s-time_%.2f-%.2f-alt_%.2f-clear.h5' % ('.', sys._getframe().f_code.co_name, solver.lower(), date_s, case_tag, time_start, time_end, alt_avg)
            fdir_tmp = f'{_fdir_tmp_}/{date_s}_{case_tag}_clear'
            fdir = f'{_fdir_general_}/lrt/{date_s}_{case_tag}_clear'
        else:
            fname_h5 = '%s/%s-%s-%s-%s-time_%.2f-%.2f-alt_%.2f-cloud.h5' % ('.', sys._getframe().f_code.co_name, solver.lower(), date_s, case_tag, time_start, time_end, alt_avg)
            fdir_tmp = f'{_fdir_tmp_}/{date_s}_{case_tag}_sat_cloud'
            fdir = f'{_fdir_general_}/lrt/{date_s}_{case_tag}_sat_cloud'

        os.makedirs(fdir_tmp, exist_ok=True)
        os.makedirs(fdir, exist_ok=True)
    
        atm_z_grid = levels
        z_list = atm_z_grid
        atm_z_grid_str = ' '.join(['%.2f' % z for z in atm_z_grid])
        
        if not os.path.exists(fname_h5) or overwrite_lrt:
            
            
            # write out the surface albedo
            #/----------------------------------------------------------------------------\#
            os.makedirs(f'{_fdir_general_}/sfc_alb', exist_ok=True)
            iter_0_fname = f'{_fdir_general_}/sfc_alb/sfc_alb_{date_s}_{time_start:.2f}_{time_end:.2f}_{alt_avg:.2f}km_iter_0.dat'
            if not os.path.exists(iter_0_fname) or overwrite_alb:

                alb_wvl = cld_leg['ssfr_zen_wvl']
                alb = cld_leg['ssfr_nad'] / cld_leg['ssfr_zen']
                alb[alb<0.0] = 0.0
                alb[alb>1.0] = 1.0    
                alb_avg = np.nanmean(alb, axis=0)
               
                with open(iter_0_fname, 'w') as f:
                    header = (f'# SSFR derived sfc albedo on {date_s}\n'
                            '# wavelength (nm)      albedo (unitless)\n'
                            )
                    # Build all profile lines in one go.
                    lines = [
                            f'{alb_wvl[i]:11.3f} '
                            f'{alb_avg[i]:12.3e}'
                            for i in range(len(alb_avg))
                            ]
                    f.write(header + "\n".join(lines))
            #\----------------------------------------------------------------------------/#
            
            
         
            flux_output = np.zeros(len(data_hsk['lon'][leg_masks[ileg]]))
            
            for ix in range(len(flux_output))[::simulation_interval]:
                flux_key_all = []
                
                if 0:#os.path.exists(f'{fdir}/flux_down_result_dict_sw_atm_corr.pk') and not new_compute:
                    print(f'Loading flux_down_result_dict_sw_atm_corr.pk from {fdir} ...')
                    with open(f'{fdir}/flux_down_result_dict_sw_atm_corr.pk', 'rb') as f:
                        flux_down_result_dict = pickle.load(f)
                    with open(f'{fdir}/flux_down_dir_result_dict_sw_atm_corr.pk', 'rb') as f:
                        flux_down_dir_result_dict = pickle.load(f)
                    with open(f'{fdir}/flux_down_diff_result_dict_sw_atm_corr.pk', 'rb') as f:
                        flux_down_diff_result_dict = pickle.load(f)
                    with open(f'{fdir}/flux_up_result_dict_sw_atm_corr.pk', 'rb') as f:
                        flux_up_result_dict = pickle.load(f)
                        
                    flux_key_all.extend(flux_down_result_dict.keys())
                    print("flux_down_result_dict keys: ", flux_down_result_dict.keys())
                    
                else:
                    flux_down_result_dict = {}
                    flux_down_dir_result_dict = {}
                    flux_down_diff_result_dict = {}
                    flux_up_result_dict = {}
                    
                    flux_down_results = []
                    flux_down_dir_results = []
                    flux_down_diff_results = []
                    flux_up_results = []
                
                flux_key = np.zeros_like(flux_output, dtype=object)
                cloudy = 0
                clear = 0
                
                # rt initialization
                #/----------------------------------------------------------------------------\#
                lrt_cfg = er3t.rtm.lrt.get_lrt_cfg()
                
                lrt_cfg['atmosphere_file'] = os.path.join(zpt_filedir, f'atm_profiles_{date_s}_{case_tag}.dat')
                # lrt_cfg['atmosphere_file'] = lrt_cfg['atmosphere_file'].replace('afglus.dat', 'afglss.dat')
                # lrt_cfg['solar_file'] = None
                lrt_cfg['solar_file'] = 'kurudz_ssfr.dat'
                # lrt_cfg['solar_file'] = lrt_cfg['solar_file'].replace('kurudz_0.1nm.dat', 'kurudz_1.0nm.dat')
                lrt_cfg['number_of_streams'] = 4
                lrt_cfg['mol_abs_param'] = 'reptran coarse'
                # lrt_cfg['mol_abs_param'] = f'reptran medium'
                input_dict_extra_general = {
                                    'crs_model': 'rayleigh Bodhaine29',
                                    # 'crs_model': 'rayleigh Nicolet',
                                    # 'crs_model': 'o3 Bogumil',
                                    # 'albedo_file': f'{_fdir_general_}/sfc_alb/sfc_alb_{date_s}.dat',
                                    'atm_z_grid': atm_z_grid_str,
                                    'wavelength_grid_file': 'wvl_grid_test.dat',
                                    # 'no_scattering':'mol',
                                    # 'no_absorption':'mol',
                                    }
                Nx_effective = len(effective_wvl)
                mute_list = ['albedo', 'wavelength', 'spline']
                #/----------------------------------------------------------------------------/#

                
                inits_rad = []
                flux_key_ix = []
                output_list = []
                tmhr_ranges_length = [len(data_hsk['lon'][leg_masks[k]]) for k in range(len(tmhr_ranges_select))]
                length_range = [0] + [np.sum(tmhr_ranges_length[:k+1]) for k in range(len(tmhr_ranges_length))]

            
                length_range_ind = bisect.bisect_left(length_range, ix)
                if length_range_ind > 0 and (ix not in length_range[1:-1]):
                    length_range_ind -= 1
                
                cld_leg = vars()[f'cld_leg_{length_range_ind}']   
                ind = ix - length_range[length_range_ind]
                        
                cot_x = cld_leg['cot'][ind]
                cwp_x = cld_leg['cwp'][ind]
                sza_x = cld_leg['sza'][ind]
                saa_x = cld_leg['saa'][ind]
                p3_alt_x = cld_leg['p3_alt'][ind]
                p3_alt_x = np.round(p3_alt_x, decimals=2)
                
                if not clear_sky:
                    input_dict_extra = copy.deepcopy(input_dict_extra_general)
                    if ((cot_x >= 0.1 and np.isfinite(cwp_x))):
                        cloudy += 1
                        

                        cer_x = cld_leg['cer'][ind]
                        cwp_x = cld_leg['cwp'][ind]
                        cth_x = cld_leg['cth'][ind]
                        cbh_x = cld_leg['cbh'][ind]
                        cgt_x = cld_leg['cgt'][ind]

                        
                        cth_ind_cld = bisect.bisect_left(z_list, cth_x)
                        cbh_ind_cld = bisect.bisect_left(z_list, cbh_x)
                        cth_ind_atm = bisect.bisect_left(atm_z_grid, cth_x)
                        cbh_ind_atm = bisect.bisect_left(atm_z_grid, cbh_x)
                        
                        fname_cld = f'{fdir_tmp}/cld_{ix:04d}.txt'
                        if os.path.exists(fname_cld):
                            os.remove(fname_cld)
                            
                        cld_cfg = er3t.rtm.lrt.get_cld_cfg()
                        cld_cfg['cloud_file'] = fname_cld
                        cld_cfg['cloud_altitude'] = z_list[cbh_ind_cld:cth_ind_cld+2]#-0.2
                        # cld_cfg['cloud_altitude'] = atm_z_grid[cbh_ind_atm:cth_ind_atm+2]#-0.2
                        cld_cfg['cloud_effective_radius']  = cer_x
                        cld_cfg['liquid_water_content'] = cwp_x*1000/(cgt_x*1000) # convert kg/m^2 to g/m^3
                        cld_cfg['cloud_optical_thickness'] = cot_x

                        dict_key_arr = np.concatenate(([cld_cfg['cloud_optical_thickness']], [cld_cfg['cloud_effective_radius']], cld_cfg['cloud_altitude'], [p3_alt_x]))
                        dict_key = '_'.join([f'{i:.3f}' for i in dict_key_arr])
                    else:
                        cld_cfg = None
                        dict_key = f'clear {p3_alt_x:.2f}'
                        clear += 1
                else:
                    cld_cfg = None
                    dict_key = f'clear {p3_alt_x:.2f}'
                    cot_x = 0.0
                    cwp_x = 0.0
                    cer_x = 0.0
                    cth_x = 0.0
                    cbh_x = 0.0
                    cgt_x = 0.0
                    input_dict_extra = copy.deepcopy(input_dict_extra_general)
                    clear += 1
                flux_key[ix] = dict_key
                
                if (cld_cfg is None) and (dict_key in flux_key_all):
                    flux_key_ix.append(dict_key)
                elif (cld_cfg is not None) and (dict_key in flux_key_all):
                    flux_key_ix.append(dict_key)
                else:
        
                    # rt setup
                    #/----------------------------------------------------------------------------\#
                    dark_sfc_init = er3t.rtm.lrt.lrt_init_mono_flx(
                            input_file  = '%s/input_%04d_dark.txt'  % (fdir_tmp, ix),
                            output_file = '%s/output_%04d_dark.txt' % (fdir_tmp, ix),
                            date        = date,
                            surface_albedo=0.0,
                            solar_zenith_angle = sza_x,
                            Nx = Nx_effective,
                            output_altitude    = [0, p3_alt_x, 'toa'],
                            input_dict_extra   = input_dict_extra_general.copy(),
                            mute_list          = ['wavelength', 'spline'],
                            lrt_cfg            = lrt_cfg,
                            cld_cfg            = cld_cfg,
                            aer_cfg            = None,
                            # output_format     = 'lambda uu edir edn',
                            )
                    
                    bright_sfc_init = er3t.rtm.lrt.lrt_init_mono_flx(
                            input_file  = '%s/input_%04d_bright.txt'  % (fdir_tmp, ix),
                            output_file = '%s/output_%04d_bright.txt' % (fdir_tmp, ix),
                            date        = date,
                            surface_albedo=1.0,
                            solar_zenith_angle = sza_x,
                            Nx = Nx_effective,
                            output_altitude    = [0, p3_alt_x, 'toa'],
                            input_dict_extra   = input_dict_extra_general.copy(),
                            mute_list          = ['wavelength', 'spline'],
                            lrt_cfg            = lrt_cfg,
                            cld_cfg            = cld_cfg,
                            aer_cfg            = None,
                            # output_format     = 'lambda uu edir edn',
                            )
                    
                    input_dict_extra_alb = copy.deepcopy(input_dict_extra)
                    input_dict_extra_alb['albedo_file'] = f'{_fdir_general_}/sfc_alb/sfc_alb_{date_s}_{time_start:.2f}_{time_end:.2f}_{alt_avg:.2f}km_iter_{iter}.dat'

                    init = er3t.rtm.lrt.lrt_init_mono_flx(
                            input_file  = '%s/input_%04d.txt'  % (fdir_tmp, ix),
                            output_file = '%s/output_%04d.txt' % (fdir_tmp, ix),
                            date        = date,
                            # surface_albedo=0.08,
                            solar_zenith_angle = sza_x,
                            Nx = Nx_effective,
                            output_altitude    = [0, p3_alt_x, 'toa'],
                            input_dict_extra   = input_dict_extra_alb.copy(),
                            mute_list          = ['albedo', 'wavelength', 'spline'],
                            lrt_cfg            = lrt_cfg,
                            cld_cfg            = cld_cfg,
                            aer_cfg            = None,
                            # output_format     = 'lambda uu edir edn',
                            )
                    #\----------------------------------------------------------------------------/#
                    inits_rad.append(copy.deepcopy(dark_sfc_init))
                    inits_rad.append(copy.deepcopy(bright_sfc_init))
                    inits_rad.append(copy.deepcopy(init))
                    output_list.extend(['%s/output_%04d_dark.txt' % (fdir_tmp, ix), '%s/output_%04d_bright.txt' % (fdir_tmp, ix), '%s/output_%04d.txt' % (fdir_tmp, ix)])
                    flux_key_all.extend([dict_key+"_dark", dict_key+"_bright", dict_key])
                    flux_key_ix.extend([dict_key+"_dark", dict_key+"_bright", dict_key])
                    
            # # Run RT
            print(f"Start running libratran calculations for {fname_h5.replace('.h5', '')} ")
            # #/----------------------------------------------------------------------------\#
            import platform
            if platform.system() == 'Darwin':
                ##### run several libratran calculations in parallel
                if len(inits_rad) > 0:
                    print('Running libratran calculations ...')
                    # check available CPU cores
                    NCPU = os.cpu_count()
                    import platform
                    if platform.system() == 'Darwin':
                        NCPU -= 2
                    er3t.rtm.lrt.lrt_run_mp(inits_rad, Ncpu=NCPU)        
                    for i in range(len(inits_rad)):
                        data = er3t.rtm.lrt.lrt_read_uvspec_flx([inits_rad[i]])
                        flux_down_result_dict[flux_key_all[i]] = np.squeeze(data.f_down)
                        flux_down_dir_result_dict[flux_key_all[i]] = np.squeeze(data.f_down_direct)
                        flux_down_diff_result_dict[flux_key_all[i]] = np.squeeze(data.f_down_diffuse)
                        flux_up_result_dict[flux_key_all[i]] = np.squeeze(data.f_up)
                        
                        flux_down_results.append(np.squeeze(data.f_down))
                        flux_down_dir_results.append(np.squeeze(data.f_down_direct))
                        flux_down_diff_results.append(np.squeeze(data.f_down_diffuse))
                        flux_up_results.append(np.squeeze(data.f_up))
            ##### run several libratran calculations one by one
            
            elif platform.system() == 'Linux':
                if len(inits_rad) > 0:
                    print('Running libratran calculations ...')
                    for i in range(len(inits_rad)):
                        if not os.path.exists(output_list[i]):
                            er3t.rtm.lrt.lrt_run(inits_rad[i])
                        else:
                            if os.path.getsize(output_list[i]) == 0:
                                er3t.rtm.lrt.lrt_run(inits_rad[i])
                        data = er3t.rtm.lrt.lrt_read_uvspec_flx([inits_rad[i]])
                        flux_down_result_dict[flux_key_all[i]] = np.squeeze(data.f_down)
                        flux_down_dir_result_dict[flux_key_all[i]] = np.squeeze(data.f_down_direct)
                        flux_down_diff_result_dict[flux_key_all[i]] = np.squeeze(data.f_down_diffuse)
                        flux_up_result_dict[flux_key_all[i]] = np.squeeze(data.f_up)
                        
                        flux_down_results.append(np.squeeze(data.f_down))
                        flux_down_dir_results.append(np.squeeze(data.f_down_direct))
                        flux_down_diff_results.append(np.squeeze(data.f_down_diffuse))
                        flux_up_results.append(np.squeeze(data.f_up))
            # #\----------------------------------------------------------------------------/#
            ###### delete input, output, cld txt files
            # for prefix in ['input', 'output', 'cld']:
            #     for filename in glob.glob(os.path.join(fdir_tmp, f'{prefix}_*.txt')):
            #         os.remove(filename)
            ###### delete atmospheric profile files for lw

            
            
            # save dict
            # status = 'wb'
            # with open(f'{fdir}/flux_down_result_dict_sw_atm_corr.pk', status) as f:
            #     pickle.dump(flux_down_result_dict, f)
            # with open(f'{fdir}/flux_down_dir_result_dict_sw_atm_corr.pk', status) as f:
            #     pickle.dump(flux_down_dir_result_dict, f)
            # with open(f'{fdir}/flux_down_diff_result_dict_sw_atm_corr.pk', status) as f:
            #     pickle.dump(flux_down_diff_result_dict, f)
            # with open(f'{fdir}/flux_up_result_dict_sw_atm_corr.pk', status) as f:
            #     pickle.dump(flux_up_result_dict, f)

            flux_down_results = np.array(flux_down_results)
            flux_down_dir_results = np.array(flux_down_dir_results)
            flux_down_diff_results = np.array(flux_down_diff_results)
            flux_up_results = np.array(flux_up_results)
            print("flux_down_results shape: ", flux_down_results.shape)
            # dark
            flux_down_dark = flux_down_results[0::3, :, :]
            flux_up_dark = flux_up_results[0::3]
            
            Fdn0_sfc = flux_down_dark[:, :, 0]
            Fup0_p3 = flux_up_dark[:, :, 1]
            Fdn0_p3 = flux_down_dark[:, :, 1]

            # bright
            flux_down_bright = flux_down_results[1::3, :, :]
            flux_up_bright = flux_up_results[1::3, :, :]
            
            Fdn1_sfc, Fup1_p3 = flux_down_bright[:, :, 0], flux_up_bright[:, :, 1]
            Fdn1_p3 = flux_down_bright[:, :, 1]
            
            Tdn = Fdn0_sfc / Fdn0_p3
            S   = 1.0 - (Fdn0_sfc / Fdn1_sfc)
            Tup = (Fup1_p3 - Fup0_p3) / Fdn1_sfc
            
            Uh = Fup0_p3
            Vh = (Fup1_p3 - Fup0_p3) * (1.0 - S)
            
            S_mean = np.nanmean(S, axis=0)
            Uh_mean = np.nanmean(Uh, axis=0)
            Vh_mean = np.nanmean(Vh, axis=0)
            
            f_S_mean = interp1d(effective_wvl, S_mean, bounds_error=False, fill_value=np.nan)
            S_mean = f_S_mean(cld_leg['ssfr_zen_wvl'])
            f_Uh_mean = interp1d(effective_wvl, Uh_mean, bounds_error=False, fill_value=np.nan)
            Uh_mean = f_Uh_mean(cld_leg['ssfr_zen_wvl'])
            f_Vh_mean = interp1d(effective_wvl, Vh_mean, bounds_error=False, fill_value=np.nan)
            Vh_mean = f_Vh_mean(cld_leg['ssfr_zen_wvl'])
            
            
            # real condition
            flux_down = flux_down_results[2::3, :, :]
            flux_up = flux_up_results[2::3, :, :]
            
            Fup_p3 = flux_up[:, :, 1]
            Fdn_p3 = flux_down[:, :, 1]

            rho_wvl = effective_wvl
            

            
            # SSFR observation
            fup_mean = np.nanmean(cld_leg['ssfr_nad'], axis=0)
            fdn_mean = np.nanmean(cld_leg['ssfr_zen'], axis=0)
            fup_std = np.nanstd(cld_leg['ssfr_nad'], axis=0)
            fdn_std = np.nanstd(cld_leg['ssfr_zen'], axis=0)
            
            alb_wvl = cld_leg['ssfr_zen_wvl']
            alb_avg = np.nanmean(cld_leg['ssfr_nad']/cld_leg['ssfr_zen'], axis=0)
            alb_avg[alb_avg<0.0] = 0.0
            alb_avg[alb_avg>1.0] = 1.0

            alpha_up = (fup_mean - Uh_mean) / (Vh_mean + S_mean * (fup_mean - Uh_mean))
            alpha_up[alpha_up<0.0] = 0.0
            alpha_up[alpha_up>1.0] = 1.0
            
            fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
            ax.plot(cld_leg['ssfr_zen_wvl'], fup_mean, '--', linewidth=1.0, color='royalblue', label='SSFR upward')
            ax.fill_between(cld_leg['ssfr_zen_wvl'],
                            fup_mean-fup_std,
                            fup_mean+fup_std, color='paleturquoise', alpha=0.75)
            ax.plot(cld_leg['ssfr_zen_wvl'], fdn_mean, '--', linewidth=1.0, color='orange', label='SSFR downward')
            ax.fill_between(cld_leg['ssfr_zen_wvl'],
                            fdn_mean-fdn_std,
                            fdn_mean+fdn_std, color='bisque', alpha=0.75)
            ax.plot(rho_wvl, np.nanmean(Fup_p3, axis=0), color='green', linewidth=2.0, label='Simulation upward')
            ax.plot(rho_wvl, np.nanmean(Fdn_p3, axis=0), color='red', linewidth=2.0, label='Simulation downward')
            ax.set_xlabel('Wavelength (nm)', fontsize=12)
            ax.set_ylabel('Flux (W m$^{-2}$ nm$^{-1}$)', fontsize=12)
            ax.set_xlim(cld_leg['ssfr_zen_wvl'][0], cld_leg['ssfr_zen_wvl'][-1])
            # ax.set_ylim([-0.05, 1.05])
            ax.legend()
            ax.set_title(f'{date_s} {time_start:.2f}-{time_end:.2f} Alt {alt_avg:.2f}km\niteration {iter}')
            fig.tight_layout()
            fig.savefig('fig/%s/%s_%s_time_%.2f-%.2f_alt-%.2fkm_flux_iteration_%d.png' % (date_s, date_s, case_tag, time_start, time_end, alt_avg, iter), bbox_inches='tight', dpi=150)
            # plt.show()
            

            
            if iter == 0:
                fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
                ax.plot(alb_wvl, alb_avg, label='SSFR upward/downward ratio')
                ax.plot(alb_wvl, alpha_up, label='updated albedo')
                ax.set_xlabel('Wavelength (nm)', fontsize=12)
                ax.set_ylabel('Albedo', fontsize=12)
                ax.set_ylim([-0.05, 1.05])
                ax.set_xlim(cld_leg['ssfr_zen_wvl'][0], cld_leg['ssfr_zen_wvl'][-1])
                ax.legend(fontsize=10)
                ax.set_title(f'{date_s} {time_start:.2f}-{time_end:.2f} Alt {alt_avg:.2f}km\niteration {iter}')
                fig.tight_layout()
                fig.savefig('fig/%s/%s_%s_time_%.2f-%.2f_alt-%.2fkm_albedo_iteration_%d.png' % (date_s, date_s, case_tag, time_start, time_end, alt_avg, iter), bbox_inches='tight', dpi=150)
                # plt.show()
            
            # write out the new surface albedo
            #/----------------------------------------------------------------------------\#
            alb_avg_update = alpha_up
            alb_avg_nonnan_first_ind = np.where(~np.isnan(alb_avg_update))[0][0]
            alb_avg_update[:alb_avg_nonnan_first_ind] = alb_avg[alb_avg_nonnan_first_ind]
            alb_avg_nonnan_last_ind = np.where(~np.isnan(alb_avg_update))[0][-1]
            alb_avg_update[alb_avg_nonnan_last_ind:] = alb_avg_update[alb_avg_nonnan_last_ind]
            with open(os.path.join(f'{_fdir_general_}/sfc_alb', f'sfc_alb_{date_s}_{time_start:.2f}_{time_end:.2f}_{alt_avg:.2f}km_iter_{iter+1:d}.dat'), 'w') as f:
                header = (f'# SSFR atmospheric corrected sfc albedo {date_s} iteration {iter+1}\n'
                        '# wavelength (nm)      albedo (unitless)\n'
                        )
                # Build all profile lines in one go.
                lines = [
                        f'{alb_wvl[i]:11.3f} '
                        f'{alb_avg_update[i]:12.3e}'
                        for i in range(len(alb_avg_update))
                        ]
                f.write(header + "\n".join(lines))
            #\----------------------------------------------------------------------------/#


    print("Finished libratran calculations.")  
    #\----------------------------------------------------------------------------/#

    return

   

def flt_trk_lrt_para(date=datetime.datetime(2024, 5, 31),
                     extent=[-60, -80, 82.4, 84.6],
                     sizes = [50, 20, 4],
                     tmhr_ranges_select=[[14.10, 14.27]],
                     sat_select=[1],
                     fname_bbr='data/bbr/ARCSIX-BBR_P3B_20240611_R0.ict',
                     fname_kt19='data/kt19/ARCSIX-KT19_P3B_20240611_R0.ict',
                     fname_LWC='data/lwc/ARCSIX-Lwc123_P3B_20240611_R1.ict',
                     modis_07_file=['./data/sat-data/20240531/MOD07_L2.A2024152.1525.061.2024153011814.hdf'],
                     case_tag='default',
                     config: Optional[FlightConfig] = None,
                     levels=None,
                     simulation_interval=3,
                     clear_sky=True,
                     lw=False,
                     manual_cloud=False,
                     manual_cloud_cth=0.0,
                     manual_cloud_cbh=0.0,
                     manual_cloud_cot=0.0,
                     manual_cloud_cwp=0.0,
                     manual_cloud_cer=0.0,
                     overwrite_atm=False,
                     overwrite_alb=False,
                     overwrite_cld=False,
                     overwrite_lrt=True,
                     new_compute=False,
                            ):

    # case specification
    #/----------------------------------------------------------------------------\#
    vname_x = 'lon'
    colors1 = ['r', 'g', 'b', 'brown']
    colors2 = ['hotpink', 'springgreen', 'dodgerblue', 'orange']
    #\----------------------------------------------------------------------------/#
    
    
    log = logging.getLogger("lrt")
    log.info("Starting processing for %s", date.strftime("%Y%m%d"))

    assert config, "FlightConfig required"
    date_s = date.strftime("%Y%m%d")

    # 1) Load all instrument & satellite metadata
    data_hsk  = load_h5(config.hsk(date_s))
    data_ssfr = load_h5(config.ssfr(date_s))
    data_hsr1 = load_h5(config.hsr1(date_s))
    data_logic= load_h5(config.logic(date_s))
    data_sat_coll = load_h5(config.sat_coll(date_s))
    head, data_kt19 = read_ict_kt19(fname_kt19, encoding="utf-8",
                                    na_values=[-9999999,-777,-888]) if fname_kt19 is not None else (None, None)
    head, data_bbr = read_ict_bbr(fname_bbr, encoding="utf-8",
                                  na_values=[-9999999,-777,-888]) if fname_bbr is not None else (None, None)
    head, data_lwc = read_ict_lwc(fname_LWC, encoding="utf-8",
                                  na_values=[-9999999,-777,-888]) if fname_LWC is not None else (None, None)

    # Satellite metadata
    sat_names, sat_tmhrs, sat_files = parse_sat(config.sat_coll(date_s))
    mid_times = np.array([np.mean(rng) for rng in tmhr_ranges_select])
    idx = closest_indices(sat_tmhrs, mid_times)
    
    
    # Solar spectrum
    f_solar = pd.read_csv('kurudz_ssfr.dat', delim_whitespace=True, comment='#', names=['wvl', 'flux'])
    wvl_solar = f_solar['wvl'].values
    flux_solar = f_solar['flux'].values/1000 # in W/m^2/nm
    flux_solar_interp = interp1d(wvl_solar, flux_solar, bounds_error=False, fill_value=0.0)


    # Build leg masks
    t_hsk = np.array(data_hsk["tmhr"])
    leg_masks = [(t_hsk>=lo)&(t_hsk<=hi) for lo,hi in tmhr_ranges_select]
    
    t_ssfr = data_ssfr['time']/3600.0  # convert to hours
    t_hsr1 = data_hsr1['time']/3600.0  # convert to hours
    t_lwc = np.array(data_lwc['tmhr'])
    t_kt19 = np.array(data_kt19['tmhr'])
    t_bbr = np.array(data_bbr['tmhr'])
    

    


    
    # create atmospheric profile
    #/----------------------------------------------------------------------------\#
    dropsonde_file_list, dropsonde_date_list, dropsonde_tmhr_list, dropsonde_lon_list, dropsonde_lat_list = dropsonde_time_loc_list()
    
    date_select = dropsonde_date_list == date.date()
    if np.sum(date_select) == 0:
        raise ValueError(f"No dropsonde data found for date {date.strftime('%Y-%m-%d')}")
    dropsonde_tmhr_array = np.array(dropsonde_tmhr_list)[date_select]
    # find the closest dropsonde time to the flight mid times
    mid_tmhr = np.array([np.mean(rng) for rng in tmhr_ranges_select])
    dropsonde_idx = closest_indices(dropsonde_tmhr_array, mid_tmhr)
    dropsonde_file = np.array(dropsonde_file_list)[date_select][dropsonde_idx[0]]
    log.info(f"Using dropsonde file: {dropsonde_file}")
    head, data_dropsonde = read_ict_dropsonde(dropsonde_file, encoding='utf-8', na_values=[-9999999, -777, -888])
    # log.info(f"Dropsonde time: {data_dropsonde['time'][0]} UTsC")
    
    
    
    zpt_filedir = f'{_fdir_general_}/zpt/{date_s}'
    os.makedirs(zpt_filedir, exist_ok=True)
    if levels is None:
        levels = np.concatenate((np.arange(0, 2.1, 0.2), 
                                np.arange(2.5, 4.1, 0.5), 
                                np.arange(5.0, 10.1, 2.5),
                                np.array([15, 20, 30., 40., 50.])))
    if 1:#not os.path.exists(os.path.join(zpt_filedir, f'atm_profiles_{date_s}_{case_tag}.dat')) or overwrite_atm:
        lon = np.mean(extent[:2])
        lat = np.mean(extent[2:])
        boundary_from_center = 0.25 # degree
        mod_lon = np.array([lon-boundary_from_center, lon+boundary_from_center])
        mod_lat = np.array([lat-boundary_from_center, lat+boundary_from_center])
        mod_extent = [mod_lon[0], mod_lon[1], mod_lat[0], mod_lat[1]]
        
        zpt_filename = f'zpt_{date_s}_{case_tag}.h5'
        
        fname_atm = f'modis_dropsonde_atm_{date_s}_{case_tag}.pk'
        
        status, ws10m = er3t.pre.atm.create_modis_dropsonde_atm(o2mix=0.20935, output_dir=zpt_filedir, output=zpt_filename, 
                                                fname_mod07=modis_07_file, dropsonde_df=data_dropsonde,
                                                levels=levels,
                                                extent=mod_extent, new_h_edge=None,sfc_T_set=None, sfc_h_to_zero=True,)
        
        atm0      = er3t.pre.atm.modis_dropsonde_arcsix_atmmod(zpt_file=f'{zpt_filedir}/{zpt_filename}',
                            fname=fname_atm, 
                            fname_co2_clim=f'{_fdir_general_}/climatology/cams73_latest_co2_conc_surface_inst_2020.nc',
                            fname_ch4_clim=f'{_fdir_general_}/climatology/cams_ch4_202005-202008.nc',
                            fname_o3_clim=f'{_fdir_general_}/climatology/ozone_merra2_202405_202408.h5',
                            fname_insitu=f'data/zpt/{date_s}/{date_s}_gases_profiles.csv',
                            date=date, extent=mod_extent,
                            overwrite=True)
    
        # write out the atmospheric profile in ascii format
        with open(os.path.join(zpt_filedir, f'atm_profiles_{date_s}_{case_tag}.dat'), 'w') as f:
            header = ('# Adjusted MODIS 07 atmospheric profile\n'
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
        
        with open(f'{zpt_filedir}/ch4_profiles_{date_s}_{case_tag}.dat', 'w') as f:  
            header = ('# Adjusted MODIS 07 atmospheric profile for ch4 only\n'
                    '#      z(km)      ch4(cm-3)\n'
                    )
            lines = [
                    f'{atm0.lev["altitude"]["data"][i]:11.3f} {atm0.lev["ch4"]["data"][i]:12.6e}'
                    for i in range(len(atm0.lev['altitude']['data']))[::-1]
                    ]
            f.write(header + "\n".join(lines))
    # =================================================================================


    # write out the surface albedo
    #/----------------------------------------------------------------------------\#
    os.makedirs(f'{_fdir_general_}/sfc_alb', exist_ok=True)
    if not os.path.exists(f'{_fdir_general_}/sfc_alb/sfc_alb_%s.dat' % date_s) or overwrite_alb:
    
        alb_file = 'data_albedo_20240607_low.h5'
        
        with h5py.File(alb_file, 'r') as f:
            # print(f['wvl'])
            alb_wvl = f['wvl'][...]
            alb_inter = f['albedo_interp'][...]
            
        alb_avg = np.nanmean(alb_inter, axis=0)
        
        
        with open(os.path.join(f'{_fdir_general_}/sfc_alb', f'sfc_alb_{date_s}.dat'), 'w') as f:
            header = ('# SSFR derived sfc albedo on 6/7\n'
                    '# wavelength (nm)      albedo (unitless)\n'
                    )
            # Build all profile lines in one go.
            lines = [
                    f'{alb_wvl[i]:11.3f} '
                    f'{alb_avg[i]:12.3e}'
                    for i in range(len(alb_avg))
                    ]
            f.write(header + "\n".join(lines))
    #\----------------------------------------------------------------------------/#
    
    xx = np.linspace(-12, 12, 241)
    yy_gaussian_vis = gaussian(xx, 0, 3.82)
    yy_gaussian_nir = gaussian(xx, 0, 5.10)
    with open(os.path.join('.', 'vis_0.1nm_0710.dat'), 'w') as f_slit:
        header = ('# SSFR Silicon slit function\n'
                    '# wavelength (nm)      relative intensity\n'
                    )
        # Build all profile lines in one go.
        lines = [
                 f'{xx[i]:11.1f} '
                 f'{yy_gaussian_vis[i]:12.5e}'
                 for i in range(len(xx))
                ]
        f_slit.write(header + "\n".join(lines))
        
    xx_wvl_grid = np.arange(360, 1990.1, 5)
    with open(os.path.join('.', 'wvl_grid_test.dat'), 'w') as f_grid:
        # Build all profile lines in one go.
        lines = [
                 f'{xx_wvl_grid[i]:11.1f} '
                 for i in range(len(xx_wvl_grid))
                ]
        f_grid.write(header + "\n".join(lines))
    
    # write out the convolved solar flux
    #/----------------------------------------------------------------------------\#
    wvl_solar_vis = np.arange(300, 950.1, 1)
    wvl_solar_nir = np.arange(951, 2500.1, 1)
    wvl_solar_coarse = np.concatenate([wvl_solar_vis, wvl_solar_nir])
    effective_wvl = wvl_solar_coarse[np.logical_and(wvl_solar_coarse >= xx_wvl_grid.min(), wvl_solar_coarse <= xx_wvl_grid.max())]
    if 1:#not os.path.exists('kurudz_ssfr.dat') or overwrite_lrt:
        df_solor = pd.read_csv('kurudz_0.1nm.dat', sep='\s+', header=None)
        wvl_solar = np.array(df_solor.iloc[:, 0])
        flux_solar = np.array(df_solor.iloc[:, 1])#/1000 # convert mW/m^2/nm to W/m^2/nm
        mask = wvl_solar <= 2500

        wvl_solar = wvl_solar[mask]
        flux_solar = flux_solar[mask]
        flux_solar_convolved = flux_solar.copy()
        
        assert (xx[1]-xx[0]) - (wvl_solar[1]-wvl_solar[0]) <1e-3

        flux_solar_convolved_vis = convolve(flux_solar, yy_gaussian_vis, mode='same') / np.sum(yy_gaussian_vis)
        flux_solar_convolved_nir = convolve(flux_solar, yy_gaussian_nir, mode='same') / np.sum(yy_gaussian_nir)
        flux_solar_convolved[wvl_solar<=950] = flux_solar_convolved_vis[wvl_solar<=950]
        flux_solar_convolved[wvl_solar>950] = flux_solar_convolved_nir[wvl_solar>950]
        
        flux_solar_convolved_coarse = np.zeros_like(wvl_solar_coarse)
        for vis_wvl_i in range(len(wvl_solar_vis)):
            ind = wvl_solar == wvl_solar_vis[vis_wvl_i]
            flux_solar_convolved_coarse[vis_wvl_i] = flux_solar_convolved_vis[ind]
        for nir_wvl_i in range(len(wvl_solar_nir)):
            ind = wvl_solar == wvl_solar_nir[nir_wvl_i]
            flux_solar_convolved_coarse[nir_wvl_i+len(wvl_solar_vis)] = flux_solar_convolved_nir[ind]
        
        with open('kurudz_ssfr.dat', 'w') as f_solar:
            header = ('# SSFR version solar flux\n'
                    '# wavelength (nm)      flux (mW/m^2/nm)\n'
                    )
            # Build all profile lines in one go.
            lines = [
                    f'{wvl_solar_coarse[i]:11.1f} '
                    f'{flux_solar_convolved_coarse[i]:12.5e}'
                    for i in range(len(wvl_solar_coarse))
                    ]
            f_solar.write(header + "\n".join(lines))
        

    

    # read satellite granule
    #/----------------------------------------------------------------------------\#
    import platform
    
    fdir_cld_obs_info = f'{_fdir_general_}/flt_cld_obs_info'
    os.makedirs(fdir_cld_obs_info, exist_ok=True)
    fname_cld_obs_info = '%s/%s_cld_obs_info_%s_%s_%s_0.pkl' % (fdir_cld_obs_info, _mission_.lower(), _platform_.lower(), date_s, case_tag)
    if 1:#not os.path.exists(fname_cld_obs_info) or overwrite_cld:      
        
        # Loop legs: load raw NC, apply cloud logic, interpolate, plot
        for i, mask in enumerate(leg_masks):
            sat_nc = config.sat_nc(date_s, sat_files[idx[i]])
                
            # read the cloud observation information
            print('Reading cloud observation information from %s ...' % sat_nc)
            f = Dataset(sat_nc, 'r')
            lon_s = f.groups['geolocation_data'].variables['longitude'][...].data
            lat_s = f.groups['geolocation_data'].variables['latitude'][...].data

            cot_s = f.groups['geophysical_data'].variables['Cloud_Optical_Thickness'][...].data
            cot_pcl_s = f.groups['geophysical_data'].variables['Cloud_Optical_Thickness_PCL'][...].data

            cer_s = f.groups['geophysical_data'].variables['Cloud_Effective_Radius'][...].data
            cer_pcl_s = f.groups['geophysical_data'].variables['Cloud_Effective_Radius_PCL'][...].data
            
            cwp_s = np.float_(f.groups['geophysical_data'].variables['Cloud_Water_Path'][...].data)
            cwp_pcl_s = np.float_(f.groups['geophysical_data'].variables['Cloud_Water_Path_PCL'][...].data)
            
            cwp_s /= 1000 # convert g/m^2 to kg/m^2
            cwp_pcl_s /= 1000 # convert g/m^2 to kg/m

            cth_s = np.float_(f.groups['geophysical_data'].variables['Cloud_Top_Height'][...].data)
            
            cth_s /= 1000 # m to km
            
            ctt_s = np.float_(f.groups['geophysical_data'].variables['Cloud_Top_Temperature'][...].data)

            ctp_s = np.float_(f.groups['geophysical_data'].variables['Cloud_Phase_Optical_Properties'][...].data)

            scan_utc_s = f.groups['scan_line_attributes'].variables['scan_start_time'][...].data
            jday_s0 = np.array([er3t.util.dtime_to_jday(datetime.datetime(1993, 1, 1, 0, 0, 0) + datetime.timedelta(seconds=scan_utc_s[i])) for i in range(scan_utc_s.size)])
            f.close()

            logic_pcl = (cot_s<0.0) & (cot_pcl_s>0.0) & (cer_s<0.0) & (cer_pcl_s>0.0)
            cot_s[logic_pcl] = cot_pcl_s[logic_pcl]
            cer_s[logic_pcl] = cer_pcl_s[logic_pcl]
            cwp_s[logic_pcl] = cwp_pcl_s[logic_pcl]
            cwp_s[cwp_s<0.0] = np.nan

            logic_0 = (ctp_s==0.0)&((cot_s<0.0)|(cer_s<0.0))
            logic_1 = (ctp_s==1.0)&((cot_s<0.0)|(cer_s<0.0))
            logic_2 = (ctp_s==2.0)&((cot_s<0.0)|(cer_s<0.0))
            logic_3 = (ctp_s==3.0)&((cot_s<0.0)|(cer_s<0.0))
            logic_4 = (ctp_s==4.0)&((cot_s<0.0)|(cer_s<0.0))

            cot_s[logic_0] = np.nan
            cer_s[logic_0] = np.nan

            cot_s[logic_1] = 0.0
            cer_s[logic_1] = 1.0

            cot_s[logic_2] = -2.0
            cer_s[logic_2] = 1.0

            cot_s[logic_3] = -3.0
            cer_s[logic_3] = 1.0

            cot_s[logic_4] = -4.0
            cer_s[logic_4] = 1.0

            cot_s[cot_s<-10.0] = np.nan
            cer_s[cot_s<-10.0] = np.nan
            
            cgt_s = np.zeros_like(cth_s)
            cgt_s[...] = np.nan
            
            # read CGT_Data.csv
            cgt_table = pd.read_csv('CGT_Data.csv')
            for j in range(len(cgt_table)):
                pixel_select_cwp_low = np.logical_and(np.logical_and(cth_s > cgt_table['cth_low'][j], cth_s <= cgt_table['cth_high'][j]),
                                                    cwp_s < cgt_table['cwp_threshold'][j]/1000)
                pixel_select_cwp_high = np.logical_and(np.logical_and(cth_s > cgt_table['cth_low'][j], cth_s <= cgt_table['cth_high'][j]),
                                                    cwp_s >= cgt_table['cwp_threshold'][j]/1000)
                cgt_s[pixel_select_cwp_low] = cgt_table['a'][j]*cwp_s[pixel_select_cwp_low] + cgt_table['b'][j]
                cgt_s[pixel_select_cwp_high] = cgt_table['a_extra'][j]*cwp_s[pixel_select_cwp_high] + cgt_table['b_extra'][j]
                
            
            ext = [0.13, 0.25, 0.39, 0.55, 0.67] # in km-1
            ctt_thresh = [0, 200, 220, 240, 260, 400]
            # thin cirrus cloud
            for j in range(len(ctt_thresh)-1):
                pixel_select_thin = np.logical_and(cot_s>0, np.logical_and(cot_s < 1., np.logical_and(ctt_s > ctt_thresh[j], ctt_s <= ctt_thresh[j+1])))
                cgt_s[pixel_select_thin] = cot_s[pixel_select_thin]/ext[j]
                

            # sys.exit()
            cgt_s[cgt_s>cth_s] = cth_s[cgt_s>cth_s]
            cbh_s = cth_s - cgt_s
            
            cot_interp = NearestNDInterpolator(list(zip(lon_s.flatten(), lat_s.flatten())), cot_s.flatten())
            cer_interp = NearestNDInterpolator(list(zip(lon_s.flatten(), lat_s.flatten())), cer_s.flatten())
            cwp_interp = NearestNDInterpolator(list(zip(lon_s.flatten(), lat_s.flatten())), cwp_s.flatten())
            cth_interp = NearestNDInterpolator(list(zip(lon_s.flatten(), lat_s.flatten())), cth_s.flatten())
            cgt_interp = NearestNDInterpolator(list(zip(lon_s.flatten(), lat_s.flatten())), cgt_s.flatten())
            cbh_interp = NearestNDInterpolator(list(zip(lon_s.flatten(), lat_s.flatten())), cbh_s.flatten())
            
            cot_s0 = cot_s.copy()
            cot_s0[...] = 255.0
            # cs = ax1.pcolormesh(lon_s, lat_s, cot_s0, cmap='gray', vmin=0.0, vmax=20.0, zorder=0, transform=ccrs.PlateCarree(), alpha=1.0)
            cot_s[cot_s<=0.0] = np.nan
            ctp_s[ctp_s==1.0] = np.nan
            cth_s[cth_s<=0.0] = np.nan
            
            proj0 = ccrs.Orthographic(
                    central_longitude=((extent[0]+extent[1])/2.0),
                    central_latitude=((extent[2]+extent[3])/2.0),
                    )
            
            # find index arrays in one go
            times_leg = t_hsk[mask]
            
            sel_ssfr, sel_hsr1, sel_lwc, sel_kt19, sel_bbr = (
                nearest_indices(t_hsk, mask, arr)
                for arr in (t_ssfr, t_hsr1, t_lwc, t_kt19, t_bbr)
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
                "ssfr_nad": data_ssfr["f_up"][sel_ssfr],
                "hsr1_wvl": data_hsr1["wvl_dn_tot"],
                "ssfr_zen_wvl": data_ssfr["wvl_dn"],
                "ssfr_nad_wvl": data_ssfr["wvl_up"],
                "lon":     data_hsk["lon"][mask],
                "lat":     data_hsk["lat"][mask],
                "sza":     data_hsk["sza"][mask],
                "saa":     data_hsk["saa"][mask],
                "p3_alt":  data_hsk["alt"][mask] / 1000.0,
                
                
            }
            if fname_bbr is not None:
                leg.update({
                    "down_ir_flux": np.array(data_bbr['DN_IR_Irrad'])[sel_bbr],
                    "up_ir_flux": np.array(data_bbr['UP_IR_Irrad'])[sel_bbr],
                    "ir_sky_T": np.array(data_bbr['IR_Sky_Temp'])[sel_bbr],
                })
            if fname_kt19 is not None:
                leg.update({
                    "ir_sfc_T": np.array(data_kt19['ir_sfc_T'])[sel_kt19],
                })
            leg.update({
                "cot": cot_interp(leg['lon'], leg['lat']),
                "cer": cer_interp(leg['lon'], leg['lat']),
                "cwp": cwp_interp(leg['lon'], leg['lat']),
                "cth": cth_interp(leg['lon'], leg['lat']),
                "cgt": cgt_interp(leg['lon'], leg['lat']),
                "cbh": cbh_interp(leg['lon'], leg['lat']),
            })

            vars()["cld_leg_%d" % i] = leg
        
            # save the cloud observation information to a pickle file
            fname_pkl = '%s/%s_cld_obs_info_%s_%s_%s_%d.pkl' % (fdir_cld_obs_info, _mission_.lower(), _platform_.lower(), date_s, case_tag, i)
            with open(fname_pkl, 'wb') as f:
                pickle.dump(vars()["cld_leg_%d" % i], f, protocol=pickle.HIGHEST_PROTOCOL)
                
                
            # # plot clouds cot, cth
            # label_size = 12
            # plt.close('all')
            # fig = plt.figure(figsize=(16, 6))
            # ax1 = fig.add_subplot(121)
            # ax2 = fig.add_subplot(122)
            # cc1 = ax1.scatter(vars()["cld_leg_%d" % i]['lon'], vars()["cld_leg_%d" % i]['lat'], c=vars()["cld_leg_%d" % i]['cot'], s=3, cmap='viridis')
            # ax1.set_title("Cloud Optical Thickness (COT)", fontsize=label_size+2)
            # ax1.set_xlabel("Longitude", fontsize=label_size)
            # ax1.set_ylabel("Latitude", fontsize=label_size)
            # # ax1.set_xlim(mod_lon.min(), mod_lon.max())
            # # ax1.set_ylim(mod_lat.min(), mod_lat.max())
            # cbar1 = fig.colorbar(cc1, ax=ax1, orientation='vertical', pad=0.02)
            # cbar1.set_label("COT", fontsize=label_size) 
            # cc2 = ax2.scatter(vars()["cld_leg_%d" % i]['lon'], vars()["cld_leg_%d" % i]['lat'], c=vars()["cld_leg_%d" % i]['cth'], s=3, cmap='viridis')
            # ax2.set_title("Cloud Top Height (CTH)", fontsize=label_size+2)
            # ax2.set_xlabel("Longitude", fontsize=label_size)
            # ax2.set_ylabel("Latitude", fontsize=label_size)
            # # ax2.set_xlim(mod_lon.min(), mod_lon.max())
            # # ax2.set_ylim(mod_lat.min(), mod_lat.max())
            # cbar2 = fig.colorbar(cc2, ax=ax2, orientation='vertical', pad=0.02)
            # cbar2.set_label("CTH (km)", fontsize=label_size) 
            # fig.suptitle(f"Clouds on {date}", fontsize=label_size+2, y=0.98)
            # fig.tight_layout()
            # plt.show()
            if platform.system() == 'Darwin':
                
                cot_s0 = cot_s.copy()
                cot_s0[...] = 255.0
                # cs = ax1.pcolormesh(lon_s, lat_s, cot_s0, cmap='gray', vmin=0.0, vmax=20.0, zorder=0, transform=ccrs.PlateCarree(), alpha=1.0)
                cot_s[cot_s<=0.0] = np.nan
                ctp_s[ctp_s==1.0] = np.nan
                
                proj0 = ccrs.Orthographic(
                        central_longitude=((extent[0]+extent[1])/2.0),
                        central_latitude=((extent[2]+extent[3])/2.0),
                        )
                plt.close('all')
                fig = plt.figure(figsize=(18, 12))
                ax1 = fig.add_subplot(111, projection=proj0)
                cs_ctp = ax1.pcolormesh(lon_s, lat_s,  ctp_s, cmap='viridis', vmin=0.0, vmax=5.0, zorder=0, transform=ccrs.PlateCarree(), alpha=0.5)
                cs_cot = ax1.pcolormesh(lon_s, lat_s,  cot_s, cmap='jet', vmin=0.0, vmax=20.0, zorder=0, transform=ccrs.PlateCarree(), alpha=0.5)

                ax1.plot(data_hsk['lon'], data_hsk['lat'], lw=2.5, color='k', transform=ccrs.PlateCarree(), zorder=1)

                color = colors1[i]

                text1 = (date + datetime.timedelta(hours=tmhr_ranges_select[i][0])).strftime('%H:%M:%S')
                text2 = (date + datetime.timedelta(hours=tmhr_ranges_select[i][1])).strftime('%H:%M:%S')
                ax1.scatter(data_hsk['lon'][mask], data_hsk['lat'][mask], color=color, s=sizes[i], lw=0.0, alpha=1.0, transform=ccrs.PlateCarree())
                # ax1.text(data_hsk['lon'][logics_select[i]][0], data_hsk['lat'][logics_select[i]][0], text1, color=color, fontsize=12, alpha=1.0, va='bottom', ha='center', transform=ccrs.PlateCarree())
                # ax1.text(data_hsk['lon'][logics_select[i]][-1], data_hsk['lat'][logics_select[i]][-1], text2, color=color, fontsize=16, alpha=1.0, va='bottom', ha='center', transform=ccrs.PlateCarree())
                # ax1.scatter(f_lon[3], f_lat[3], s=5, marker='^', color='orange')
                # ax1.axvline(-52.3248, color='b', lw=1.0, alpha=1.0, zorder=0)
                # ax1.axvline(-51.7540, color='g', lw=1.0, alpha=1.0, zorder=0)
                # ax1.axvline(-51.3029, color='r', lw=1.0, alpha=1.0, zorder=0)

                sat_select_text = os.path.basename(sat_nc).replace('.nc', '').replace('CLDPROP_L2_', '')
                sat_ = sat_select_text.split('.')[0].replace('_', ' ')
                sat_utc = sat_select_text.split('.')[2]
                title_text = f'{sat_} {sat_utc}\n' + \
                    f'Flight track {text1} - {text2} UTC'
                ax1.set_title(title_text, fontsize=24, y=1.01, color='k')
                ax1.coastlines(resolution='10m', color='gray', lw=0.5)
                g1 = ax1.gridlines(lw=0.5, color='gray', draw_labels=True, ls='-')
                g1.xlocator = FixedLocator(np.arange(-180, 181, 5.0))
                g1.ylocator = FixedLocator(np.arange(-90.0, 89.9, 0.2))
                g1.top_labels = False
                g1.right_labels = False
                

                ax1.set_extent(extent, crs=ccrs.PlateCarree())
                
                cbar1 = fig.colorbar(cs_cot, ax=ax1, orientation='vertical', pad=0.05, aspect=50, shrink=0.8)
                cbar1.set_label('Cloud Optical Thickness', fontsize=16)
                cbar1.ax.tick_params(labelsize=14)
                
                # cbar2 = fig.colorbar(cs_ctp, ax=ax1, orientation='vertical', pad=0.05, aspect=50, shrink=0.8)
                # cbar2.set_label('Cloud Phase', fontsize=16)
                # cbar2.ax.tick_params(labelsize=14)
                #\--------------------------------------------------------------/#

                # save figure
                #/--------------------------------------------------------------\#
                fig.subplots_adjust(hspace=0.3, wspace=0.3)
                _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                fig.savefig('fig/%s/%s_%s_sat_%d_leg_%d.png' % (date_s, date_s, case_tag, idx[i], i), bbox_inches='tight', metadata=_metadata)
                #\--------------------------------------------------------------/#
        # sys.exit()
    else:
        print('Loading cloud observation information from %s ...' % fname_cld_obs_info)
        for i in range(len(tmhr_ranges_select)):
            fname_pkl = '%s/%s_cld_obs_info_%s_%s_%s_%d.pkl' % (fdir_cld_obs_info, _mission_.lower(), _platform_.lower(), date_s, case_tag, i)
            with open(fname_pkl, 'rb') as f:
                vars()[f"cld_leg_{i}"] = pickle.load(f)   
    
    sys.exit()
    solver = 'lrt'

    if not lw:
        if clear_sky:
            fname_h5 = '%s/%s-%s-%s-%s-clear.h5' % ('.', sys._getframe().f_code.co_name, solver.lower(), date_s, case_tag)
            fdir_tmp = f'{_fdir_tmp_}/{date_s}_{case_tag}_clear'
            fdir = f'{_fdir_general_}/lrt/{date_s}_{case_tag}_clear'
        else:
            if manual_cloud:
                fname_h5 = '%s/%s-%s-%s-%s-manual_cloud.h5' % ('.', sys._getframe().f_code.co_name, solver.lower(), date_s, case_tag)
                fdir_tmp = f'{_fdir_tmp_}/{date_s}_{case_tag}_manual_cloud'
                fdir = f'{_fdir_general_}/lrt/{date_s}_{case_tag}_manual_cloud'
            else:
                fname_h5 = '%s/%s-%s-%s-%s-sat_cloud.h5' % ('.', sys._getframe().f_code.co_name, solver.lower(), date_s, case_tag)
                fdir_tmp = f'{_fdir_tmp_}/{date_s}_{case_tag}_sat_cloud'
                fdir = f'{_fdir_general_}/lrt/{date_s}_{case_tag}_sat_cloud'
    else:
        fdir_lw_zpt_tmp = f'{_fdir_tmp_}/{date_s}_{case_tag}_lw_zpt'
        os.makedirs(fdir_lw_zpt_tmp, exist_ok=True)
        if clear_sky:
            fname_h5 = '%s/%s-%s-%s-%s-clear-lw.h5' % ('.', sys._getframe().f_code.co_name, solver.lower(), date_s, case_tag)
            fdir_tmp = f'{_fdir_tmp_}/{date_s}_{case_tag}_clear-lw'
            fdir = f'{_fdir_general_}/lrt/{date_s}_{case_tag}_clear-lw'
        else:
            if manual_cloud:
                fname_h5 = '%s/%s-%s-%s-%s-manual_cloud-lw.h5' % ('.', sys._getframe().f_code.co_name, solver.lower(), date_s, case_tag)
                fdir_tmp = f'{_fdir_tmp_}/{date_s}_{case_tag}_manual_cloud-lw'
                fdir = f'{_fdir_general_}/lrt/{date_s}_{case_tag}_manual_cloud-lw'
            else:
                fname_h5 = '%s/%s-%s-%s-%s-sat_cloud-lw.h5' % ('.', sys._getframe().f_code.co_name, solver.lower(), date_s, case_tag)
                fdir_tmp = f'{_fdir_tmp_}/{date_s}_{case_tag}_sat_cloud-lw'
                fdir = f'{_fdir_general_}/lrt/{date_s}_{case_tag}_sat_cloud-lw'
  
    os.makedirs(fdir_tmp, exist_ok=True)
    os.makedirs(fdir, exist_ok=True)
    
    if not os.path.exists(fname_h5) or overwrite_lrt:
         
        atm_z_grid = levels
        z_list = atm_z_grid
        atm_z_grid_str = ' '.join(['%.2f' % z for z in atm_z_grid])
        

        flux_output = np.zeros(np.sum([len(data_hsk['lon'][logics_select[k]]) for k in range(len(tmhr_ranges_select))]))
        

        flux_key_all = []
        
        if os.path.exists(f'{fdir}/flux_down_result_dict_sw.pk') and not new_compute:
            print(f'Loading flux_down_result_dict_sw.pk from {fdir} ...')
            with open(f'{fdir}/flux_down_result_dict_sw.pk', 'rb') as f:
                flux_down_result_dict = pickle.load(f)
            with open(f'{fdir}/flux_down_dir_result_dict_sw.pk', 'rb') as f:
                flux_down_dir_result_dict = pickle.load(f)
            with open(f'{fdir}/flux_down_diff_result_dict_sw.pk', 'rb') as f:
                flux_down_diff_result_dict = pickle.load(f)
            with open(f'{fdir}/flux_up_result_dict_sw.pk', 'rb') as f:
                flux_up_result_dict = pickle.load(f)
                
            flux_key_all.extend(flux_down_result_dict.keys())
            print("flux_down_result_dict keys: ", flux_down_result_dict.keys())
            
        else:
            flux_down_result_dict = {}
            flux_down_dir_result_dict = {}
            flux_down_diff_result_dict = {}
            flux_up_result_dict = {}
            
        
        flux_key = np.zeros_like(flux_output, dtype=object)
        cloudy = 0
        clear = 0
        
        # rt initialization
        #/----------------------------------------------------------------------------\#
        lrt_cfg = er3t.rtm.lrt.get_lrt_cfg()
        
        if not lw:
            lrt_cfg['atmosphere_file'] = os.path.join(zpt_filedir, f'atm_profiles_{date_s}_{case_tag}.dat')
            # lrt_cfg['atmosphere_file'] = lrt_cfg['atmosphere_file'].replace('afglus.dat', 'afglss.dat')
            # lrt_cfg['solar_file'] = None
            lrt_cfg['solar_file'] = 'kurudz_ssfr.dat'
            # lrt_cfg['solar_file'] = lrt_cfg['solar_file'].replace('kurudz_0.1nm.dat', 'kurudz_1.0nm.dat')
            lrt_cfg['number_of_streams'] = 4
            lrt_cfg['mol_abs_param'] = 'reptran coarse'
            # lrt_cfg['mol_abs_param'] = f'reptran medium'
            input_dict_extra_general = {
                                'crs_model': 'rayleigh Bodhaine29',
                                # 'crs_model': 'rayleigh Nicolet',
                                # 'crs_model': 'o3 Bogumil',
                                'albedo_file': f'{_fdir_general_}/sfc_alb/sfc_alb_{date_s}.dat',
                                'atm_z_grid': atm_z_grid_str,
                                'wavelength_grid_file': 'wvl_grid_test.dat',
                                # 'no_scattering':'mol',
                                # 'no_absorption':'mol',
                                }
            Nx_effective = len(effective_wvl)
            mute_list = ['albedo', 'wavelength', 'spline']
        else:
            # lrt_cfg['atmosphere_file'] = os.path.join(zpt_filedir, f'atm_profiles_{date_s}_{case_tag}.dat')
            lrt_cfg['number_of_streams'] = 4
            lrt_cfg['mol_abs_param'] = 'reptran coarse'
            # ch4_file = os.path.join(zpt_filedir, f'ch4_profiles_{date_s}_{case_tag}.dat')
            input_dict_extra_general = {
                                'source': 'thermal',
                                'albedo_add': '0',
                                'atm_z_grid': atm_z_grid_str,
                                # 'mol_file': f'CH4 {ch4_file}',
                                # 'wavelength_grid_file': 'wvl_grid_thermal.dat',
                                'wavelength_add' : '4500 42000',
                                'output_process': 'integrate',
                                }
            Nx_effective = 1 # integrate over all wavelengths
            mute_list = ['albedo', 'wavelength', 'spline', 'source solar', 'atmosphere_file']
        #/----------------------------------------------------------------------------/#

        
        inits_rad = []
        flux_key_ix = []
        output_list = []
        tmhr_ranges_length = [len(data_hsk['lon'][logics_select[k]]) for k in range(len(tmhr_ranges_select))]
        length_range = [0] + [np.sum(tmhr_ranges_length[:k+1]) for k in range(len(tmhr_ranges_length))]
        print("len(flux_output): ", len(flux_output))
        print("length_range: ", length_range)
        for ix in range(len(flux_output))[::simulation_interval]:
            length_range_ind = bisect.bisect_left(length_range, ix)
            if length_range_ind > 0 and (ix not in length_range[1:-1]):
                length_range_ind -= 1
            
            cld_leg = vars()[f'cld_leg_{length_range_ind}']
                    
            ind = ix - length_range[length_range_ind]
                    
            # print(f"ix {ix}, index {ind}")
            # print(f"altitude {np.round(cld_leg['p3_alt'][ind], decimals=2)} ...")
                
            cot_x = cld_leg['cot'][ind]
            cwp_x = cld_leg['cwp'][ind]
            sza_x = cld_leg['sza'][ind]
            saa_x = cld_leg['saa'][ind]
            p3_alt_x = cld_leg['p3_alt'][ind]
            p3_alt_x = np.round(p3_alt_x, decimals=2)
            if fname_kt19 is not None:
                sfc_T_kt19 = cld_leg['ir_sfc_T'][ind] + +273.15 # convert to Kelvin
                sfc_T_kt19 = np.round(sfc_T_kt19, decimals=1)
            else:
                sfc_T_kt19 = None
            if not clear_sky:
                input_dict_extra = copy.deepcopy(input_dict_extra_general)
                if ((cot_x >= 0.1 and np.isfinite(cwp_x))) or manual_cloud:
                    cloudy += 1
                    
                    if not manual_cloud:
                        cer_x = cld_leg['cer'][ind]
                        cwp_x = cld_leg['cwp'][ind]
                        cth_x = cld_leg['cth'][ind]
                        cbh_x = cld_leg['cbh'][ind]
                        cgt_x = cld_leg['cgt'][ind]
                    else:
                        # manual cloud properties
                        cer_x = manual_cloud_cer
                        cwp_x = manual_cloud_cwp
                        cth_x = manual_cloud_cth
                        cbh_x = manual_cloud_cbh
                        cot_x = manual_cloud_cot
                        cgt_x = cth_x-cbh_x
                    
                    cth_ind_cld = bisect.bisect_left(z_list, cth_x)
                    cbh_ind_cld = bisect.bisect_left(z_list, cbh_x)
                    cth_ind_atm = bisect.bisect_left(atm_z_grid, cth_x)
                    cbh_ind_atm = bisect.bisect_left(atm_z_grid, cbh_x)
                    
                    fname_cld = f'{fdir_tmp}/cld_{ix:04d}.txt'
                    if os.path.exists(fname_cld):
                        os.remove(fname_cld)
                        
                    cld_cfg = er3t.rtm.lrt.get_cld_cfg()
                    cld_cfg['cloud_file'] = fname_cld
                    cld_cfg['cloud_altitude'] = z_list[cbh_ind_cld:cth_ind_cld+2]#-0.2
                    # cld_cfg['cloud_altitude'] = atm_z_grid[cbh_ind_atm:cth_ind_atm+2]#-0.2
                    cld_cfg['cloud_effective_radius']  = cer_x
                    cld_cfg['liquid_water_content'] = cwp_x*1000/(cgt_x*1000) # convert kg/m^2 to g/m^3
                    cld_cfg['cloud_optical_thickness'] = cot_x
                    # print(cer_x, cwp_x, cot_x, cld_cfg['cloud_altitude'])
                    
                    if not lw:
                        dict_key_arr = np.concatenate(([cld_cfg['cloud_optical_thickness']], [cld_cfg['cloud_effective_radius']], cld_cfg['cloud_altitude'], [p3_alt_x]))
                    else:
                        dict_key_arr = np.concatenate(([cld_cfg['cloud_optical_thickness']], [cld_cfg['cloud_effective_radius']], cld_cfg['cloud_altitude'], [p3_alt_x], [sfc_T_kt19]))
                    dict_key = '_'.join([f'{i:.3f}' for i in dict_key_arr])
                    # input_dict_extra['wc_properties'] = 'mie interpolate'
                else:
                    cld_cfg = None
                    if not lw:
                        dict_key = f'clear {p3_alt_x:.2f}'
                    else:
                        dict_key = f'clear {p3_alt_x:.2f} {sfc_T_kt19:.1f}'
                    clear += 1
            else:
                cld_cfg = None
                if not lw:
                    dict_key = f'clear {p3_alt_x:.2f}'
                else:
                    dict_key = f'clear {p3_alt_x:.2f} {sfc_T_kt19:.1f}'
                cot_x = 0.0
                cwp_x = 0.0
                cer_x = 0.0
                cth_x = 0.0
                cbh_x = 0.0
                cgt_x = 0.0
                input_dict_extra = copy.deepcopy(input_dict_extra_general)
                clear += 1
            flux_key[ix] = dict_key
            
            if (cld_cfg is None) and (dict_key in flux_key_all):
                flux_key_ix.append(dict_key)
            elif (cld_cfg is not None) and (dict_key in flux_key_all):
                flux_key_ix.append(dict_key)
            else:
                
                if lw:
                    # generate the new atmospheric profile with kt19 surface temperature for each simulation
                    if sfc_T_kt19 is None:
                        raise ValueError("sfc_T_kt19 must be provided for longwave simulations.")
                    if levels is None:
                        levels = np.concatenate((np.arange(0, 2.1, 0.2), 
                                                np.arange(2.5, 4.1, 0.5), 
                                                np.arange(5.0, 10.1, 2.5),
                                                np.array([15, 20, 30., 40., 50.])))
                    if not os.path.exists(os.path.join(fdir_lw_zpt_tmp, f'atm_profiles_{date_s}_{case_tag}_sfcT_{sfc_T_kt19:.1f}K.dat')) or overwrite_atm:
                        lon = np.mean(extent[:2])
                        lat = np.mean(extent[2:])
                        boundary_from_center = 0.25 # degree
                        mod_lon = np.array([lon-boundary_from_center, lon+boundary_from_center])
                        mod_lat = np.array([lat-boundary_from_center, lat+boundary_from_center])
                        mod_extent = [mod_lon[0], mod_lon[1], mod_lat[0], mod_lat[1]]
                        
                        zpt_filename = f'zpt_{date_s}_{case_tag}_sfcT_{sfc_T_kt19:.1f}K.h5'
                        
                        fname_atm = f'modis_dropsonde_atm_{date_s}_{case_tag}_sfcT_{sfc_T_kt19:.1f}K.pk'
                        
                        status, ws10m = er3t.pre.atm.create_modis_dropsonde_atm(o2mix=0.20935, output_dir=fdir_lw_zpt_tmp, output=zpt_filename, 
                                                                fname_mod07=modis_07_file, dropsonde_df=data_dropsonde,
                                                                levels=levels,
                                                                extent=mod_extent, new_h_edge=None,sfc_T_set=sfc_T_kt19, sfc_h_to_zero=True, plot=False,)
                        
                        atm0      = er3t.pre.atm.modis_dropsonde_atmmod(zpt_file=f'{fdir_lw_zpt_tmp}/{zpt_filename}',
                                            fname=f'{fdir_lw_zpt_tmp}/{fname_atm}', 
                                            fname_co2_clim=f'{_fdir_general_}/climatology/cams73_latest_co2_conc_surface_inst_2020.nc',
                                            fname_o3_clim=f'{_fdir_general_}/climatology/ozone_merra2_202405_202408.h5',
                                            date=date, extent=mod_extent,
                                            overwrite=True, plot=False,)
                    
                        # write out the atmospheric profile in ascii format
                        with open(os.path.join(fdir_lw_zpt_tmp, f'atm_profiles_{date_s}_{case_tag}_{sfc_T_kt19:.1f}K.dat'), 'w') as f:
                            header = ('# Adjusted MODIS 07 atmospheric profile\n'
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
                        
                        with open(f'{fdir_lw_zpt_tmp}/ch4_profiles_{date_s}_{case_tag}_{sfc_T_kt19:.1f}K.dat', 'w') as f:  
                            header = ('# Adjusted MODIS 07 atmospheric profile for ch4 only\n'
                                    '#      z(km)      ch4(cm-3)\n'
                                    )
                            lines = [
                                    f'{atm0.lev["altitude"]["data"][i]:11.3f} {atm0.lev["ch4"]["data"][i]:12.6e}'
                                    for i in range(len(atm0.lev['altitude']['data']))[::-1]
                                    ]
                            f.write(header + "\n".join(lines))
                    input_dict_extra['atmosphere_file_add'] = os.path.join(fdir_lw_zpt_tmp, f'atm_profiles_{date_s}_{case_tag}_{sfc_T_kt19:.1f}K.dat')
                    ch4_file = os.path.join(fdir_lw_zpt_tmp, f'ch4_profiles_{date_s}_{case_tag}_{sfc_T_kt19:.1f}K.dat')
                    input_dict_extra['mol_file'] = f'CH4 {ch4_file}'
                
                
                
                # rt setup
                #/----------------------------------------------------------------------------\#
                
                init = er3t.rtm.lrt.lrt_init_mono_flx(
                        input_file  = '%s/input_%04d.txt'  % (fdir_tmp, ix),
                        output_file = '%s/output_%04d.txt' % (fdir_tmp, ix),
                        date        = date,
                        # surface_albedo=0.08,
                        solar_zenith_angle = sza_x,
                        # wavelength         = wavelength,
                        Nx = Nx_effective,
                        output_altitude    = [p3_alt_x, 'toa', 0],
                        input_dict_extra   = input_dict_extra.copy(),
                        mute_list          = mute_list,
                        lrt_cfg            = lrt_cfg,
                        cld_cfg            = cld_cfg,
                        aer_cfg            = None,
                        # output_format     = 'lambda uu edir edn',
                        )
                #\----------------------------------------------------------------------------/#
                inits_rad.append(copy.deepcopy(init))
                output_list.append('%s/output_%04d.txt' % (fdir_tmp, ix))
                flux_key_all.append(dict_key)
                flux_key_ix.append(dict_key)
                    
        print('len(inits_rad): ', len(inits_rad))
        print("flux_key_all: ", flux_key_all)
        print("flux_key_ix set: ", set(flux_key_ix))
        print("flux_key_all length: ", len(flux_key_all))
        print("flux_key_ix length: ", len(flux_key_ix))
        print("len set(flux_key_ix): ", len(set(flux_key_ix)))
        print("set(flux_key_ix) == set(flux_key_all): ", set(flux_key_ix) == set(flux_key_all))
        # sys.exit()
        # # Run RT
        print(f"Start running libratran calculations for {fname_h5.replace('.h5', '')} ")
        # #/----------------------------------------------------------------------------\#
        import platform
        if platform.system() == 'Darwin':
            ##### run several libratran calculations in parallel
            if len(inits_rad) > 0:
                print('Running libratran calculations ...')
                # check available CPU cores
                NCPU = os.cpu_count()
                import platform
                if platform.system() == 'Darwin':
                    NCPU -= 2
                er3t.rtm.lrt.lrt_run_mp(inits_rad, Ncpu=NCPU)        
                for i in range(len(inits_rad)):
                    data = er3t.rtm.lrt.lrt_read_uvspec_flx([inits_rad[i]])
                    flux_down_result_dict[flux_key_all[i]] = np.squeeze(data.f_down)
                    flux_down_dir_result_dict[flux_key_all[i]] = np.squeeze(data.f_down_direct)
                    flux_down_diff_result_dict[flux_key_all[i]] = np.squeeze(data.f_down_diffuse)
                    flux_up_result_dict[flux_key_all[i]] = np.squeeze(data.f_up)
        ##### run several libratran calculations one by one
        
        elif platform.system() == 'Linux':
            if len(inits_rad) > 0:
                print('Running libratran calculations ...')
                for i in range(len(inits_rad)):
                    if not os.path.exists(output_list[i]):
                        er3t.rtm.lrt.lrt_run(inits_rad[i])
                    else:
                        if os.path.getsize(output_list[i]) == 0:
                            er3t.rtm.lrt.lrt_run(inits_rad[i])
                    data = er3t.rtm.lrt.lrt_read_uvspec_flx([inits_rad[i]])
                    flux_down_result_dict[flux_key_all[i]] = np.squeeze(data.f_down)
                    flux_down_dir_result_dict[flux_key_all[i]] = np.squeeze(data.f_down_direct)
                    flux_down_diff_result_dict[flux_key_all[i]] = np.squeeze(data.f_down_diffuse)
                    flux_up_result_dict[flux_key_all[i]] = np.squeeze(data.f_up)
        # #\----------------------------------------------------------------------------/#
        ###### delete input, output, cld txt files
        # for prefix in ['input', 'output', 'cld']:
        #     for filename in glob.glob(os.path.join(fdir_tmp, f'{prefix}_*.txt')):
        #         os.remove(filename)
        ###### delete atmospheric profile files for lw
        if lw:
            if platform.system() == 'Darwin':
                for filename in glob.glob(os.path.join(fdir_lw_zpt_tmp, f'atm_profiles*.dat')):
                    os.remove(filename)
                for filename in glob.glob(os.path.join(fdir_lw_zpt_tmp, f'ch4_profiles*.dat')):
                    os.remove(filename)
            for filename in glob.glob(os.path.join(fdir_lw_zpt_tmp, f'modis_dropsonde_atm_*.pk')):
                os.remove(filename)
            for filename in glob.glob(os.path.join(fdir_lw_zpt_tmp, f'zpt_*.h5')):
                os.remove(filename)
            
        
        # save dict
        status = 'wb'
        with open(f'{fdir}/flux_down_result_dict_sw.pk', status) as f:
            pickle.dump(flux_down_result_dict, f)
        with open(f'{fdir}/flux_down_dir_result_dict_sw.pk', status) as f:
            pickle.dump(flux_down_dir_result_dict, f)
        with open(f'{fdir}/flux_down_diff_result_dict_sw.pk', status) as f:
            pickle.dump(flux_down_diff_result_dict, f)
        with open(f'{fdir}/flux_up_result_dict_sw.pk', status) as f:
            pickle.dump(flux_up_result_dict, f)


        
        flux_output_t = np.zeros(len(range(len(flux_output))))
        f_tmhr = np.zeros(len(flux_output))
        f_lon = np.zeros(len(flux_output_t))
        f_lat = np.zeros(len(flux_output_t))
        f_alt = np.zeros(len(flux_output_t))
        f_sza = np.zeros(len(flux_output_t))
        f_cth = np.zeros(len(flux_output_t))
        f_cbh = np.zeros(len(flux_output_t))
        f_cot = np.zeros(len(flux_output_t))
        f_cwp = np.zeros(len(flux_output_t))
        f_cer = np.zeros(len(flux_output_t))
        f_cgt = np.zeros(len(flux_output_t))
        f_ssfr_zen = np.zeros((len(flux_output_t), len(data_ssfr['zen/wvl'])))
        f_ssfr_nad = np.zeros((len(flux_output_t), len(data_ssfr['nad/wvl'])))
        f_hsr1_total = np.zeros((len(flux_output_t), 401))
        f_hsr1_diff = np.zeros((len(flux_output_t), 401))
        f_twp = np.zeros(len(flux_output_t))
        f_lwc_1 = np.zeros(len(flux_output_t))
        f_lwc_2 = np.zeros(len(flux_output_t))
        f_bbr_up = np.zeros(len(flux_output_t))
        f_bbr_down = np.zeros(len(flux_output_t))
        f_bbr_sky_T = np.zeros(len(flux_output_t))
        f_kt19 = np.zeros(len(flux_output_t))
        
        f_down_1d = np.zeros((len(flux_output_t), Nx_effective, 3))
        f_down_dir_1d = np.zeros((len(flux_output_t), Nx_effective, 3))
        f_down_diff_1d = np.zeros((len(flux_output_t), Nx_effective, 3))
        f_up_1d = np.zeros((len(flux_output_t), Nx_effective, 3))
        
        for f_array in [f_tmhr, f_lon, f_lat, f_alt, f_sza,
                        f_cth, f_cbh, f_cot, f_cwp, f_cer, f_cgt, f_ssfr_zen, f_ssfr_nad, 
                        f_hsr1_total, f_hsr1_diff,
                        f_twp, f_lwc_1, f_lwc_2, 
                        f_bbr_up, f_bbr_down, f_bbr_sky_T, f_kt19,
                        f_down_1d, f_down_dir_1d, f_down_diff_1d, f_up_1d]:
            f_array[...] = np.nan

        
        for ix in range(len(flux_output))[::simulation_interval]:
            length_range_ind = bisect.bisect_left(length_range, ix)
            if length_range_ind > 0 and (ix not in length_range[1:-1]):
                length_range_ind -= 1
            
            cld_leg = vars()[f'cld_leg_{length_range_ind}']
                    
            ind = ix - length_range[length_range_ind]
                    
            f_down_1d[ix] = flux_down_result_dict[flux_key[ix]]
            f_down_dir_1d[ix] = flux_down_dir_result_dict[flux_key[ix]]
            f_down_diff_1d[ix] = flux_down_diff_result_dict[flux_key[ix]]
            f_up_1d[ix] = flux_up_result_dict[flux_key[ix]]
                
                
        for ix in range(len(flux_output)):
            length_range_ind = bisect.bisect_left(length_range, ix)
            if length_range_ind > 0 and (ix not in length_range[1:-1]):
                length_range_ind -= 1
            
            cld_leg = vars()[f'cld_leg_{length_range_ind}']
                    
            ind = ix - length_range[length_range_ind]
            
            f_tmhr[ix] = cld_leg['tmhr'][ind]
            f_lon[ix] = cld_leg['lon'][ind]
            f_lat[ix] = cld_leg['lat'][ind]
            f_alt[ix] = cld_leg['p3_alt'][ind]
            f_sza[ix] = cld_leg['sza'][ind]
            f_ssfr_zen[ix] = cld_leg['ssfr_zen'][ind]
            f_ssfr_nad[ix] = cld_leg['ssfr_nad'][ind]
            f_hsr1_total[ix] = cld_leg['hsr1_total'][ind]
            f_hsr1_diff[ix] = cld_leg['hsr1_dif'][ind]
            if fname_LWC is not None:
                f_twp[ix] = cld_leg['twc'][ind]
                f_lwc_1[ix] = cld_leg['lwc_1'][ind]
                f_lwc_2[ix] = cld_leg['lwc_2'][ind]
            if fname_bbr is not None:
                f_bbr_up[ix] = cld_leg['up_ir_flux'][ind]
                f_bbr_down[ix] = cld_leg['down_ir_flux'][ind]
                f_bbr_sky_T[ix] = cld_leg['ir_sky_T'][ind]
            if fname_kt19 is not None:
                f_kt19[ix] = cld_leg['ir_sfc_T'][ind]
                
            if not clear_sky:
                if not manual_cloud:
                        f_cer[ix] = cld_leg['cer'][ind]
                        f_cwp[ix] = cld_leg['cwp'][ind]
                        f_cth[ix] = cld_leg['cth'][ind]
                        f_cbh[ix] = cld_leg['cbh'][ind]
                        f_cgt[ix] = cld_leg['cgt'][ind]
                        f_cot[ix] = cld_leg['cot'][ind]
                else:
                    # manual cloud properties
                    f_cer[ix] = manual_cloud_cer
                    f_cwp[ix] = manual_cloud_cwp
                    f_cth[ix] = manual_cloud_cth
                    f_cbh[ix] = manual_cloud_cbh
                    f_cot[ix] = manual_cloud_cot
                    f_cgt[ix] = manual_cloud_cth-manual_cloud_cbh
            else:
                f_cth[ix] = 0.0
                f_cbh[ix] = 0.0
                f_cot[ix] = 0.0
                f_cwp[ix] = 0.0
                f_cgt[ix] = 0.0
                f_cer[ix] = 0.0
            
                
        # save rad_2d results
        with h5py.File(fname_h5, 'w') as f:
            f.create_dataset('tmhr', data=f_tmhr)
            f.create_dataset('lon', data=f_lon)
            f.create_dataset('lat', data=f_lat)
            f.create_dataset('alt', data=f_alt)
            f.create_dataset('sza', data=f_sza)
            f.create_dataset('f_down', data=f_down_1d)
            f.create_dataset('f_down_dir', data=f_down_dir_1d)
            f.create_dataset('f_down_diff', data=f_down_diff_1d)
            f.create_dataset('f_up', data=f_up_1d)
            f.create_dataset('cth', data=f_cth)
            f.create_dataset('cbh', data=f_cbh)
            f.create_dataset('cot', data=f_cot)
            f.create_dataset('cwp', data=f_cwp)
            f.create_dataset('cgt', data=f_cgt)
            f.create_dataset('cer', data=f_cer)
            f.create_dataset('ssfr_zen', data=f_ssfr_zen)
            f.create_dataset('ssfr_nad', data=f_ssfr_nad)
            f.create_dataset('ssfr_nad_wvl', data=data_ssfr['nad/wvl'])
            f.create_dataset('ssfr_zen_wvl', data=data_ssfr['zen/wvl'])
            f.create_dataset('ssfr_toa0', data=data_ssfr['zen/toa0'])
            f.create_dataset('hsr1_toa0', data=data_hsr1['tot/toa0'])
            f.create_dataset('hsr1_total', data=f_hsr1_total)
            f.create_dataset('hsr1_diff', data=f_hsr1_diff)
            f.create_dataset('hsr1_wvl', data=data_hsr1['wvl_dn_tot'])
            f.create_dataset('twc', data=f_twp)
            f.create_dataset('lwc_1', data=f_lwc_1)
            f.create_dataset('lwc_2', data=f_lwc_2)
            f.create_dataset('bbr_up', data=f_bbr_up)
            f.create_dataset('bbr_down', data=f_bbr_down)
            f.create_dataset('bbr_sky_T', data=f_bbr_sky_T)
            f.create_dataset('kt19', data=f_kt19)
            
    else:
        print('Loading existing libratran results ...')
        with h5py.File(fname_h5, 'r') as f:
            f_tmhr = f['tmhr'][...]
            f_lon = f['lon'][...]
            f_lat = f['lat'][...]
            f_alt = f['alt'][...]
            f_sza = f['sza'][...]
            f_down_1d = f['f_down'][...]
            f_down_dir_1d = f['f_down_dir'][...]
            f_down_diff_1d = f['f_down_diff'][...]
            f_up_1d = f['f_up'][...]
            f_cth = f['cth'][...]
            f_cbh = f['cbh'][...]
            f_cot = f['cot'][...]
            f_cwp = f['cwp'][...]
            f_cgt = f['cgt'][...]
            f_ssfr_zen = f['ssfr_zen'][...]
            f_ssfr_nad = f['ssfr_nad'][...]
            data_ssfr_wvl_zen = np.array(f['ssfr_zen_wvl'])
            data_ssfr_wvl_nad = np.array(f['ssfr_nad_wvl'])
            data_ssfr_toa0 = np.array(f['ssfr_toa0'])
            data_hsr1_toa0 = np.array(f['hsr1_toa0'])
            data_hsr1_total = np.array(f['hsr1_total'])
            data_hsr1_diff = np.array(f['hsr1_diff'])
            data_hsr1_wvl = np.array(f['hsr1_wvl'])
            f_twp = f['twc'][...]
            f_lwc_1 = f['lwc_1'][...]
            f_lwc_2 = f['lwc_2'][...]
            f_bbr_up = f['bbr_up'][...]
            f_bbr_down = f['bbr_down'][...]
            f_bbr_sky_T = f['bbr_sky_T'][...]
            f_kt19 = f['kt19'][...]
        
        #############

    print("Finished libratran calculations.")  
    #\----------------------------------------------------------------------------/#

    return


if __name__ == '__main__':

    # figure_01_arcsix1_flight_tracks_all()
    # figure_cop_wall_20240605()
    # figure_cop_wall_20240607()
    # figure_cop_wall_20240611()
    # figure_cop_flt_trk_20240605()
    # figure_cop_flt_trk_20240607()
    # figure_cop_flt_trk_20240611()
    # sys.exit()
    
    dir_fig = './fig'
    os.makedirs(dir_fig, exist_ok=True)
    
    config = FlightConfig(mission='ARCSIX',
                            platform='P3B',
                            data_root=_fdir_data_,
                            sat_root_mac='/Volumes/argus/field/arcsix/sat-data',
                            sat_root_linux='/pl/active/vikas-arcsix/yuch8913/arcsix/data/sat-data',)
    
    
    # gases_insitu(date=datetime.datetime(2024, 6, 5), 
    #              gas_dir='data/gases', 
    #              config=config,
    #              plot=True,
    #              time_select_range=[[15.04, 15.28]]
    #              )
    
    # for date in [datetime.datetime(2024, 5, 31),
    #              datetime.datetime(2024, 6, 3),
    #              datetime.datetime(2024, 6, 5),
    #              datetime.datetime(2024, 6, 6),
    #              datetime.datetime(2024, 6, 7),
    #              datetime.datetime(2024, 6, 11),
    #              datetime.datetime(2024, 6, 13),
    #              ]:

    #     gases_insitu(date=date, 
    #                 gas_dir='data/gases', 
    #                 config=config,
    #                 plot=True,
    #                 )
    
    # flt_trk_flux_R0(date=datetime.datetime(2024, 5, 31),
    #             tmhr_ranges_select=[[14.10, 14.50]],)
    
    # flt_trk_flux_R0(date=datetime.datetime(2024, 5, 31),
    #             tmhr_ranges_select=[[14.10, 14.70]],)
    
    # flt_trk_flux_R0(date=datetime.datetime(2024, 5, 31),
    #             tmhr_ranges_select=[[14.10, 15.10]],)
    
    # flt_trk_flux_R0(date=datetime.datetime(2024, 5, 31),
    #             tmhr_ranges_select=[[14.10, 15.50]],)
    
    # flt_trk_flux_R0(date=datetime.datetime(2024, 5, 31),
    #             tmhr_ranges_select=[[16.20, 16.72]],)
    
    # flt_trk_flux_R0(date=datetime.datetime(2024, 6, 5),
    #             tmhr_ranges_select=[[15.55, 16.32]],)
    
    # flt_trk_flux_R0(date=datetime.datetime(2024, 6, 5),
    #             tmhr_ranges_select=[[13.2, 13.81]],)
    
    # sys.exit()
    
    """flt_trk_flux_R0(date=datetime.datetime(2024, 5, 31),
                tmhr_ranges_select=[[13.97, 14.04], [14.12 , 14.28], [14.31, 14.42], [14.52 , 14.60], [14.72, 14.85], [14.88, 15.03], ],)
    
    flt_trk_flux_R0(date=datetime.datetime(2024, 5, 31),
                tmhr_ranges_select=[[15.88, 16.22], [16.31, 16.46], [16.52, 16.72]],)
    
    flt_trk_flux_R0(date=datetime.datetime(2024, 6, 5),
                tmhr_ranges_select=[[13.2, 13.35], [13.41, 13.45], [13.57, 13.69], [13.72, 13.80]],)
    
    flt_trk_flux_R0(date=datetime.datetime(2024, 6, 5),
                tmhr_ranges_select=[[14.27, 14.49], [14.59, 15.02]],)
    
    flt_trk_flux_R0(date=datetime.datetime(2024, 6, 6),
                tmhr_ranges_select=[[16.39 , 16.45], [16.53, 16.62]],)"""
    
    # flt_trk_flux_R0(date=datetime.datetime(2024, 6, 6),
    #             tmhr_ranges_select=[[16.54, 16.94]],)
    
    # flt_trk_flux_R0(date=datetime.datetime(2024, 6, 13),
    #             tmhr_ranges_select=[[16.78, 17.00]],)
    # sys.exit()
    
    # marli_flt_trk_lrt_para(date=datetime.datetime(2024, 6, 5),
    #                  extent=[-44, -58, 83.3, 84.1],
    #                  sizes = [50, 20, 4],
    #                  tmhr_ranges_select=[[15.36, 15.60], [16.32, 16.60], [16.78, 16.85]],
    #                  fname_marli='data/marli/ARCSIX-MARLi_P3B_20240605_R0.cdf',
    #                  fname_kt19='data/kt19/ARCSIX-MetNav-KT19-10Hz_P3B_20240605_R0.ict',
    #                  case_tag='marli_test',
    #                  config=config)
    
    # marli_flt_trk_lrt_para(date=datetime.datetime(2024, 6, 5),
    #                  extent=[-44, -58, 83.3, 84.1],
    #                  sizes = [50, 20, 4],
    #                  tmhr_ranges_select=[[15.36, 15.60], [16.32, 16.60], [16.78, 16.85]],
    #                  fname_marli='data/marli/ARCSIX-MARLi_P3B_20240605_R0.cdf',
    #                  fname_kt19='data/kt19/ARCSIX-MetNav-KT19-10Hz_P3B_20240605_R0.ict',
    #                  case_tag='marli_test',
    #                  config=config)
    
    # marli_flt_trk_lrt_para(date=datetime.datetime(2024, 6, 13),
    #                  extent=[-10, -20, 81.9, 82.7],
    #                  sizes = [50, 20, 4],
    #                  tmhr_ranges_select=[[13.5, 13.6], [13.67, 13.78]],
    #                  fname_marli='data/marli/ARCSIX-MARLi_P3B_20240613_R0.cdf',
    #                  fname_kt19='data/kt19/ARCSIX-MetNav-KT19-10Hz_P3B_20240613_R0.ict',
    #                  case_tag='marli_0613_1',
    #                  config=config)
    
    # marli_flt_trk_lrt_para(date=datetime.datetime(2024, 6, 13),
    #                  extent=[-50, -66, 78.0, 80.0],
    #                  sizes = [50, 20, 4],
    #                  tmhr_ranges_select=[[11.52, 11.94]],
    #                  fname_marli='data/marli/ARCSIX-MARLi_P3B_20240613_R0.cdf',
    #                  fname_kt19='data/kt19/ARCSIX-MetNav-KT19-10Hz_P3B_20240613_R0.ict',
    #                  case_tag='marli_0613_1',
    #                  config=config)
    
    # marli_flt_trk_lrt_para(date=datetime.datetime(2024, 6, 11),
    #                  extent=[-60, -70, 85.2, 86.0],
    #                  sizes = [50, 20, 4],
    #                  tmhr_ranges_select=[[13.08, 13.21], [13.33, 13.43]],
    #                  fname_marli='data/marli/ARCSIX-MARLi_P3B_20240611_R0.cdf',
    #                  fname_kt19='data/kt19/ARCSIX-MetNav-KT19-10Hz_P3B_20240611_R0.ict',
    #                  case_tag='marli_test_0611_1',
    #                  config=config)
    
    # marli_flt_trk_lrt_para(date=datetime.datetime(2024, 6, 7),
    #                  extent=[-60, -70, 78.0, 79.6],
    #                  sizes = [50, 20, 4],
    #                  tmhr_ranges_select=[[13.8, 13.87], [13.92, 14.0]],
    #                  fname_marli='data/marli/ARCSIX-MARLi_P3B_20240607_R0.cdf',
    #                  fname_kt19='data/kt19/ARCSIX-MetNav-KT19-10Hz_P3B_20240607_R0.ict',
    #                  case_tag='marli_test_0607_1',
    #                  config=config)
    
    # marli_flt_trk_lrt_para(date=datetime.datetime(2024, 5, 28),
    #                  extent=[-42, -52, 84.2, 85.9],
    #                  sizes = [50, 20, 4],
    #                  tmhr_ranges_select=[[15.66, 15.96]],
    #                  fname_marli='data/marli/ARCSIX-MARLi_P3B_20240528_R0.cdf',
    #                  fname_kt19='data/kt19/ARCSIX-MetNav-KT19-10Hz_P3B_20240528_R0.ict',
    #                  case_tag='marli_test_0528_1',
    #                  config=config)
    
    # marli_flt_trk_lrt_para(date=datetime.datetime(2024, 5, 28),
    #                  extent=[-40, -52, 84.2, 85.9],
    #                  sizes = [50, 20, 4],
    #                  tmhr_ranges_select=[[15.05, 15.12]],
    #                  fname_marli='data/marli/ARCSIX-MARLi_P3B_20240528_R0.cdf',
    #                  fname_kt19='data/kt19/ARCSIX-MetNav-KT19-10Hz_P3B_20240528_R0.ict',
    #                  case_tag='marli_test_0528_1',
    #                  config=config)
    
    
    # sys.exit()
    
    # flt_trk_lwc(date=datetime.datetime(2024, 6, 5),
    #             tmhr_ranges_select=[[13, 17], [14.5, 16.1], [15.04, 15.28]],
    #             output_lwp_alt=[False, False, True],
    #             fname_LWC=f'{_fdir_general_}/lwc/ARCSIX-Lwc123_P3B_20240605_R1.ict',
    #             fname_cloud_micro_2DGRAY50=f'{_fdir_general_}/cloud_prob/2DGRAY50/ARCSIX-2DGRAY50_P3B_20240611105230_R1.ict',
    #             fname_cloud_micro_FCDP=f'{_fdir_general_}/cloud_prob/FCDP/ARCSIX-FCDP_P3B_20240605_R1.ict',
    #             timeoff_FCDP=0.005,
    #             config=config
    #             )
    
    # flt_trk_lwc(date=datetime.datetime(2024, 6, 11),
    #             tmhr_ranges_select=[[13.9111, 15.7139], [14.03, 14.075]],
    #             output_lwp_alt=[False, True],
    #             fname_LWC=f'{_fdir_general_}/lwc/ARCSIX-Lwc123_P3B_20240611_R1.ict',
    #             fname_cloud_micro_2DGRAY50=f'{_fdir_general_}/cloud_prob/2DGRAY50/ARCSIX-2DGRAY50_P3B_20240611105230_R1.ict',
    #             fname_cloud_micro_FCDP=f'{_fdir_general_}/cloud_prob/FCDP/ARCSIX-FCDP_P3B_20240611_R1.ict',
    #             timeoff_FCDP=0.002,
    #             config=config
    #             )
    
    
    # flt_trk_lwc(date=datetime.datetime(2024, 6, 7),
    #             tmhr_ranges_select=[[15.34, 16.27], [15.765, 15.795]],
    #             output_lwp_alt=[False, True],
    #             fname_LWC=f'{_fdir_general_}/lwc/ARCSIX-Lwc123_P3B_20240607_R1.ict',
    #             fname_cloud_micro_2DGRAY50=f'{_fdir_general_}/cloud_prob/2DGRAY50/ARCSIX-2DGRAY50_P3B_20240607104243_R1.ict',
    #             fname_cloud_micro_FCDP=f'{_fdir_general_}/cloud_prob/FCDP/ARCSIX-FCDP_P3B_20240607_R1.ict',
    #             timeoff_FCDP=0.002,
    #             config=config
    #             )
    
    # flt_trk_lwc(date=datetime.datetime(2024, 6, 13),
    #             tmhr_ranges_select=[[14.0, 16.5], [14.92, 15.28], [15.03, 15.11], [15.8, 16.1], [15.88, 15.94], [15.84, 15.98]],
    #             output_lwp_alt=[False, False, True, False, True, True],
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
    #             timeoff_FCDP=0.046,
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
    
    
    # sys.exit()
    

    # flt_trk_lrt_para(date=datetime.datetime(2024, 6, 7),
    #                 extent = [-55, -40, 83.4, 85.2],
    #                 sizes = [50, 20, 4],
    #                 tmhr_ranges_select=[[15.3400, 15.7583], [15.8403, 16.2653]],
    #                 sat_select=[1],
    #                 fname_radiosonde=f'{_fdir_general_}/radiosonde/arcsix-THAAO-RSxx_SONDE_20240607191800_RA.ict',
    #                 fname_dropsonde=f'{_fdir_general_}/dropsonde/ARCSIX-AVAPS_G3_20240607_R0/ARCSIX-AVAPS_G3_20240607160915_R0.ict',
    #                 fname_bbr=f'{_fdir_general_}/bbr/ARCSIX-BBR_P3B_20240607_R0.ict',
    #                 fname_kt19=f'{_fdir_general_}/kt19/ARCSIX-MetNav-KT19-10Hz_P3B_20240607_R0.ict',
    #                 fname_LWC=f'{_fdir_general_}/lwc/ARCSIX-Lwc123_P3B_20240607_R1.ict',
    #                 modis_07_file=[f'{_fdir_general_}/sat-data/20240607/MYD07_L2.A2024159.1520.061.2024160161210.hdf'],
    #                 simulation_interval=30,
    #                 levels=np.concatenate((np.array([0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 1.0]),
    #                                         np.array([1.25, 1.5, 2.0, 2.5, 3.0, 4.0]), 
    #                                         np.arange(5.0, 10.1, 2.5),
    #                                         np.array([15, 20, 30., 40., 45.]))),
    #                 case_tag='cloudy_track_2',
    #                 lw=False,
    #                 clear_sky=False,
    #                 manual_cloud=True,
    #                 manual_cloud_cer=8.0,
    #                 manual_cloud_cwp=0.0229,
    #                 manual_cloud_cth=0.47,
    #                 manual_cloud_cbh=0.25,
    #                 manual_cloud_cot=4.3,
    #                 overwrite_atm=True,
    #                 overwrite_alb=False,
    #                 overwrite_cld=True,
    #                 overwrite_lrt=True,
    #                 new_compute=False,)
    
    # flt_trk_lrt_para(date=datetime.datetime(2024, 6, 13),
    #                 extent=[-39, -47, 83.3, 84.1],
    #                 sizes = [50, 20, 4],
    #                 tmhr_ranges_select=[[15.84, 15.88], [15.94, 15.98]],
    #                 sat_select=[1],
    #                 fname_radiosonde=f'{_fdir_general_}/radiosonde/arcsix-THAAO-RSxx_SONDE_20240613183800_RA.ict',
    #                 fname_dropsonde=f'{_fdir_general_}/dropsonde/ARCSIX-AVAPS_G3_20240613_R0/ARCSIX-AVAPS_G3_20240613151255_R0.ict',
    #                 fname_bbr=f'{_fdir_general_}/bbr/ARCSIX-BBR_P3B_20240613_R0.ict',
    #                 fname_kt19=f'{_fdir_general_}/kt19/ARCSIX-MetNav-KT19-10Hz_P3B_20240613_R0.ict',
    #                 fname_LWC=f'{_fdir_general_}/lwc/ARCSIX-Lwc123_P3B_20240613_R1.ict',
    #                 modis_07_file=[f'{_fdir_general_}/sat-data/20240613/MYD07_L2.A2024165.1610.061.2024166155733.hdf'],
    #                 simulation_interval=1,
    #                 case_tag='cloudy_track_1',
    #                 levels=np.concatenate((np.arange(0.0, 1.01, 0.1),
    #                                         np.array([1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0]), 
    #                                         np.arange(5.0, 10.1, 2.5),
    #                                         np.array([15, 20, 30., 40., 45.]))),
    #                 lw=False,
    #                 clear_sky=False,
    #                 manual_cloud=True,
    #                 manual_cloud_cer=14.4,
    #                 manual_cloud_cwp=0.06013,
    #                 manual_cloud_cth=0.945,
    #                 manual_cloud_cbh=0.344,
    #                 manual_cloud_cot=6.26,
    #                 overwrite_atm=True,
    #                 overwrite_alb=False,
    #                 overwrite_cld=True,
    #                 overwrite_lrt=True,
    #                 new_compute=True,
    #                 )
    

    # flt_trk_lrt_para(date=datetime.datetime(2024, 6, 3),
    #                             extent=[-42, -48, 83.5, 84.5],
    #                           sizes = [50, 20, 4],
    #                           tmhr_ranges_select=[[14.72, 14.86], [14.95, 15.09]],
    #                           sat_select=[1],
    #                           fname_radiosonde=f'{_fdir_general_}/radiosonde/arcsix-THAAO-RSxx_SONDE_20240603180200_RA.ict',
    #                           fname_dropsonde=f'{_fdir_general_}/dropsonde/ARCSIX-AVAPS_G3_20240603_R0/ARCSIX-AVAPS_G3_20240603142310_R0.ict',
    #                           fname_bbr=f'{_fdir_general_}/bbr/ARCSIX-BBR_P3B_20240603_R0.ict',
    #                           fname_kt19=f'{_fdir_general_}/kt19/ARCSIX-MetNav-KT19-10Hz_P3B_20240603_R0.ict',
    #                           fname_LWC=f'{_fdir_general_}/lwc/ARCSIX-Lwc123_P3B_20240603_R1.ict',
    #                           modis_07_file=[f'{_fdir_general_}/sat-data/20240603/MYD07_L2.A2024155.1555.061.2024156171946.hdf'],
    #                           case_tag='cloudy_track_1',
    #                           simulation_interval=1,
    #                 levels=np.concatenate((np.arange(0.0, 1.1, 0.1),
    #                                        np.arange(1.05, 2.51, 0.05),
    #                                         np.array([3.0, 3.5, 4.0]), 
    #                                         np.arange(5.0, 10.1, 2.5),
    #                                         np.array([15, 20, 30., 40., 45.]))),
    #                 lw=False,
    #                 clear_sky=False,
    #                 manual_cloud=True,
    #                 manual_cloud_cer=5.9,
    #                 manual_cloud_cwp=0.1012,
    #                 manual_cloud_cth=2.23,
    #                 manual_cloud_cbh=0.33,
    #                 manual_cloud_cot=25.78,
    #                 overwrite_atm=True,
    #                 overwrite_alb=False,
    #                 overwrite_cld=True,
    #                 overwrite_lrt=True,
    #                 new_compute=True,
    #                 )
    
    
    # flt_trk_lrt_para(date=datetime.datetime(2024, 6, 6),
    #                 extent=[-9, -17, 82.7, 83.7],
    #                 sizes = [50, 20, 4],
    #                 tmhr_ranges_select=[[13.99, 14.18], [14.26, 14.46]],
    #                 sat_select=[6, 7],
    #                 fname_bbr=f'{_fdir_general_}/bbr/ARCSIX-BBR_P3B_20240606_R0.ict',
    #                 fname_kt19=f'{_fdir_general_}/kt19/ARCSIX-MetNav-KT19-10Hz_P3B_20240606_R0.ict',
    #                 fname_LWC=f'{_fdir_general_}/lwc/ARCSIX-Lwc123_P3B_20240606_R1.ict',
    #                 modis_07_file=[f'{_fdir_general_}/sat-data/20240606/MYD07_L2.A2024158.1620.061.2024159154912.hdf'],
    #                 case_tag='cloudy_track_2_3out',
    #                 config=config,
    #                 simulation_interval=30,
    #                 levels=np.concatenate((np.arange(0.0, 1.61, 0.1),
    #                                         np.array([1.8, 2.0, 2.5, 3.0, 4.0]), 
    #                                         np.arange(5.0, 10.1, 2.5),
    #                                         np.array([15, 20, 30., 40., 45.]))),
    #                 lw=False,
    #                 clear_sky=False,
    #                 manual_cloud=True,
    #                 manual_cloud_cer=6.9,
    #                 manual_cloud_cwp=0.0231,
    #                 manual_cloud_cth=0.3,
    #                 manual_cloud_cbh=0.101,
    #                 manual_cloud_cot=5.01,
    #                 overwrite_atm=False,
    #                 overwrite_alb=False,
    #                 overwrite_cld=True,
    #                 overwrite_lrt=True,
    #                 new_compute=True,
    #               )
    
    
    # flt_trk_lrt_para(date=datetime.datetime(2024, 6, 11),
    #                 extent = [-72, -50, 83.4, 84.4],
    #                 sizes = [50, 20, 4],
    #                 tmhr_ranges_select=[[13.9111, 14.3417], [15.3528, 15.7139]],
    #                 sat_select=[8],
    #                 fname_radiosonde=f'{_fdir_general_}/radiosonde/arcsix-THAAO-RSxx_SONDE_20240611190300_RA.ict',
    #                 fname_dropsonde=f'{_fdir_general_}/dropsonde/ARCSIX-AVAPS_G3_20240611_R0/ARCSIX-AVAPS_G3_20240611143225_R0.ict',
    #                 fname_LWC=f'{_fdir_general_}/lwc/ARCSIX-Lwc123_P3B_20240611_R1.ict',
    #                 modis_07_file=[f'{_fdir_general_}/sat-data/20240611/MYD07_L2.A2024163.1450.061.2024164151334.hdf'],
    #                 simulation_interval=3,
    #                 clear_sky=True,
    #                 manual_cloud=True,
    #                 manual_cloud_cer=4.3,
    #                 manual_cloud_cwp=0.477,
    #                 manual_cloud_cth=3.01,
    #                 manual_cloud_cbh=3.00,
    #                 manual_cloud_cot=0.17,
    #                 overwrite_atm=True,
    #                 overwrite_alb=False,
    #                 overwrite_cld=True,
    #                 overwrite_lrt=True)
    
    # flt_trk_lrt_para(date=datetime.datetime(2024, 5, 31),
    #                             extent=[-60, -80, 82.4, 84.6],
    #                           sizes = [50, 20, 4],
    #                           tmhr_ranges_select=[[14.10, 14.27], [16.49, 16.72]],
    #                           sat_select=[1],
    #                           fname_radiosonde=f'{_fdir_general_}/radiosonde/arcsix-THAAO-RSxx_SONDE_20240531183300_RA.ict',
    #                           fname_dropsonde=f'{_fdir_general_}/dropsonde/ARCSIX-AVAPS_G3_20240531_R0/ARCSIX-AVAPS_G3_20240531142150_R0.ict',
    #                           fname_bbr=f'{_fdir_general_}/bbr/ARCSIX-BBR_P3B_20240531_R0.ict',
    #                           fname_kt19=f'{_fdir_general_}/kt19/ARCSIX-MetNav-KT19-10Hz_P3B_20240531_R0.ict',
    #                           fname_LWC=f'{_fdir_general_}/lwc/ARCSIX-Lwc123_P3B_20240611_R1.ict',
    #                           modis_07_file=[f'{_fdir_general_}/sat-data/20240531/MOD07_L2.A2024152.1525.061.2024153011814.hdf'],
    #                           case_tag='clear_sky_track_1',
    #                           simulation_interval=1,
    #                           clear_sky=True,
    #                           lw=True,
    #                 overwrite_atm=False,
    #                 overwrite_alb=False,
    #                 overwrite_cld=True,
    #                 overwrite_lrt=True,
    #                 new_compute=True,
    #                 )

    
    # flt_trk_lrt_para(date=datetime.datetime(2024, 6, 13),
    #                             extent=[-61, -68, 81.2, 81.8],
    #                           sizes = [50, 20, 4],
    #                           tmhr_ranges_select=[[16.78, 16.85], [16.91, 17.00]],
    #                           sat_select=[3],
    #                           fname_radiosonde=f'{_fdir_general_}/radiosonde/arcsix-THAAO-RSxx_SONDE_20240613183800_RA.ict',
    #                           fname_dropsonde=f'{_fdir_general_}/dropsonde/ARCSIX-AVAPS_G3_20240613_R0/ARCSIX-AVAPS_G3_20240613151255_R0.ict',
    #                           fname_bbr=f'{_fdir_general_}/bbr/ARCSIX-BBR_P3B_20240613_R0.ict',
    #                           fname_kt19=f'{_fdir_general_}/kt19/ARCSIX-MetNav-KT19-10Hz_P3B_20240613_R0.ict',
    #                           fname_LWC=f'{_fdir_general_}/lwc/ARCSIX-Lwc123_P3B_20240613_R1.ict',
    #                           modis_07_file=[f'{_fdir_general_}/sat-data/20240613/MYD07_L2.A2024165.1610.061.2024166155733.hdf'],
    #                           simulation_interval=1,
    #                           case_tag='clear_sky_track_1',
    #                         #   levels=np.concatenate((np.array([0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 1.0]),
    #                         #                 np.array([1.5, 2.0, 3.0, 4.0]), 
    #                         #                 np.arange(5.0, 10.1, 2.5),
    #                         #                 np.array([15, 20, 30., 40., 45.]))),
    #                           clear_sky=True,
    #                           lw=True,
    #                 overwrite_atm=True,
    #                 overwrite_alb=False,
    #                 overwrite_cld=True,
    #                 overwrite_lrt=False,
    #                 new_compute=False,
    #                 )

    # flt_trk_atm_corr(date=datetime.datetime(2024, 6, 5),
    #                 extent=[-68, -42, 83.0, 84.1],
    #                 tmhr_ranges_select=[[15.55, 15.9292], [16.0431, 16.32]],
    #                 modis_07_file=[f'{_fdir_general_}/sat-data/20240605/MYD07_L2.A2024157.1540.061.2024158183620.hdf'],
    #                 case_tag='clear_sky_track_atm_corr',
    #                 config=config,
    #                 simulation_interval=1,
    #                 clear_sky=True,
    #                 overwrite_atm=False,
    #                 overwrite_alb=False,
    #                 overwrite_lrt=True,
    #                 manual_cloud=False,
    #                 manual_cloud_cer=0.0,
    #                 manual_cloud_cwp=0.0,
    #                 manual_cloud_cth=0.0,
    #                 manual_cloud_cbh=0.0,
    #                 manual_cloud_cot=0.0,
    #                 iter=1,
    #                 )
    
    flt_trk_atm_corr(date=datetime.datetime(2024, 6, 6),
                    extent=[-58, -50, 83.2, 84.0],
                    tmhr_ranges_select=[[16.54, 16.62], [16.85, 16.94]],
                    modis_07_file=[f'{_fdir_general_}/sat-data/20240606/MYD07_L2.A2024158.1620.061.2024159154912.hdf'],
                    case_tag='clear_sky_track_1_atm_corr',
                    config=config,
                    simulation_interval=1,
                    clear_sky=True,
                    overwrite_atm=False,
                    overwrite_alb=False,
                    overwrite_lrt=True,
                    manual_cloud=False,
                    manual_cloud_cer=0.0,
                    manual_cloud_cwp=0.0,
                    manual_cloud_cth=0.0,
                    manual_cloud_cbh=0.0,
                    manual_cloud_cot=0.0,
                    iter=1,
                    )
    
    
    
    sys.exit()
    
    
    flt_trk_lrt_para(date=datetime.datetime(2024, 6, 5),
                                extent=[-68, -42, 83.0, 84.1],
                              sizes = [50, 20, 4],
                              tmhr_ranges_select=[[15.55, 15.9292], [16.0431, 16.32]],
                              sat_select=[1],
                              fname_bbr=f'{_fdir_general_}/bbr/ARCSIX-BBR_P3B_20240605_R0.ict',
                              fname_kt19=f'{_fdir_general_}/kt19/ARCSIX-MetNav-KT19-10Hz_P3B_20240605_R0.ict',
                              fname_LWC=f'{_fdir_general_}/lwc/ARCSIX-Lwc123_P3B_20240605_R1.ict',
                              modis_07_file=[f'{_fdir_general_}/sat-data/20240605/MYD07_L2.A2024157.1540.061.2024158183620.hdf'],
                              case_tag='clear_sky_track_1',
                              config=config,
                              simulation_interval=1,
                              clear_sky=True,
                              lw=True,
                    overwrite_atm=False,
                    overwrite_alb=False,
                    overwrite_cld=True,
                    overwrite_lrt=True,
                    new_compute=True,
                    )

    
    # flt_trk_lrt_para(date=datetime.datetime(2024, 6, 6),
    #                             extent=[-58, -50, 83.2, 84.0],
    #                           sizes = [50, 20, 4],
    #                           tmhr_ranges_select=[[16.54, 16.62], [16.85, 16.94]],
    #                           sat_select=[1],
    #                           fname_radiosonde=f'{_fdir_general_}/radiosonde/arcsix-THAAO-RSxx_SONDE_20240606234800_RA.ict',
    #                           fname_dropsonde=f'{_fdir_general_}/dropsonde/ARCSIX-AVAPS_G3_20240606_R0/ARCSIX-AVAPS_G3_20240606161914_R0.ict',
    #                           fname_bbr=f'{_fdir_general_}/bbr/ARCSIX-BBR_P3B_20240606_R0.ict',
    #                           fname_kt19=f'{_fdir_general_}/kt19/ARCSIX-MetNav-KT19-10Hz_P3B_20240606_R0.ict',
    #                           fname_LWC=f'{_fdir_general_}/lwc/ARCSIX-Lwc123_P3B_20240606_R1.ict',
    #                           modis_07_file=[f'{_fdir_general_}/sat-data/20240606/MYD07_L2.A2024158.1620.061.2024159154912.hdf'],
    #                           case_tag='clear_sky_track_1',
    #                           simulation_interval=1,
    #                           clear_sky=True,
    #                           lw=True,
    #                 overwrite_atm=False,
    #                 overwrite_alb=False,
    #                 overwrite_cld=True,
    #                 overwrite_lrt=True,
    #                 new_compute=False,
    #                 )

    pass
