import os
import sys
import glob
import datetime
import copy
import multiprocessing as mp
from collections import OrderedDict
# from tqdm import tqdm
import h5py
from pyhdf.SD import SD, SDC
from netCDF4 import Dataset
import numpy as np
from scipy import interpolate
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path as mpl_path
import matplotlib.image as mpl_img
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib import rcParams, ticker
from matplotlib.ticker import FixedLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
# import cartopy.crs as ccrs
# mpl.use('Agg')


import er3t

def figure_time_series():

    # read simulation
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname = '/data/hong/mygit/sim/data/flt_sim_0745.0000nm_20230805.h5'
    f = h5py.File(fname, 'r')
    tmhr = f['tmhr'][...]
    ang_pit = f['ang_pit'][...]
    ang_rol = f['ang_rol'][...]

    logic = (np.abs(ang_pit)<=5.0) & (np.abs(ang_rol)<=2.5)

    tmhr = tmhr[logic]
    alt = f['alt'][...][logic]
    cth = f['cth'][...][logic]

    f_dn_mca_3d = f['f-down_mca-3d'][...][logic]
    f_dn_mca_ipa = f['f-down_mca-ipa'][...][logic]
    f_dn_hsr1 = f['f-down_hsr1'][...][logic]

    f_dn_dif_mca_3d = f['f-down-diffuse_mca-3d'][...][logic]
    f_dn_dif_hsr1 = f['f-down-diffuse_hsr1'][...][logic]

    f.close()
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    plot = True
    if plot:
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        # fig.suptitle('Figure')
        # plot1
        #╭──────────────────────────────────────────────────────────────╮#
        ax1 = fig.add_subplot(111)
        ax1.scatter(tmhr, f_dn_hsr1, s=2, c='k', lw=0.0)
        ax1.scatter(tmhr, f_dn_mca_3d, s=2, c='b', lw=0.0)
        ax1.scatter(tmhr, f_dn_mca_ipa, s=2, c='r', lw=0.0)

        ax2 = ax1.twinx()
        ax2.plot(tmhr, alt, color='green')
        ax2.plot(tmhr, cth, color='purple')
        # ax1.set_xlim((0, 1))
        # ax1.set_ylim((0, 1))
        # ax1.set_xlabel('X')
        # ax1.set_ylabel('Y')
        # ax1.set_title('Plot1')
        # ax1.xaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
        # ax1.yaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
        #╰──────────────────────────────────────────────────────────────╯#
        # save figure
        #╭──────────────────────────────────────────────────────────────╮#
        fig.subplots_adjust(hspace=0.35, wspace=0.35)
        _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
        fname_fig = '%s.png' % (_metadata_['Function'],)
        plt.savefig(fname_fig, bbox_inches='tight', metadata=_metadata_, transparent=False)
        #╰──────────────────────────────────────────────────────────────╯#
        plt.show()
        sys.exit()
        plt.close(fig)
        plt.clf()
    #╰────────────────────────────────────────────────────────────────────────────╯#

if __name__ == '__main__':

    figure_time_series()
    pass
