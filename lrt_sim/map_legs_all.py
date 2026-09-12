import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import numpy as np
import cartopy.crs as ccrs
from matplotlib.ticker import FixedLocator
from datetime import datetime
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle
import cartopy.feature as cfeature
from datetime import datetime, timedelta, date
# from georadii.util import read_hsk_arcsix
import h5py

def julian_day_to_datetime(jd: float) -> datetime:
    """
    Convert a Julian Day Number (which starts at noon) into a UTC datetime.
    Algorithm: Fliegel–Van Flandern.
    
    Parameters
    ----------
    jd : float
        Julian Day (e.g. 2459725.5 for 2022-01-01 00:00 UTC).
    
    Returns
    -------
    datetime.datetime
        Corresponding UTC date & time.
    """
    # shift so that day starts at midnight
    jd += 0.5
    Z = int(jd)
    F = jd - Z
    if Z < 2299161:
        A = Z
    else:
        alpha = int((Z - 1867216.25) / 36524.25)
        A = Z + 1 + alpha - alpha//4
    B = A + 1524
    C = int((B - 122.1) / 365.25)
    D = int(365.25 * C)
    E = int((B - D) / 30.6001)

    day = B - D - int(30.6001 * E) + F
    month = E - 1 if E < 14 else E - 13
    year = C - 4716 if month > 2 else C - 4715

    # extract time from fractional day
    day_int = int(day)
    frac = day - day_int
    hours = frac * 24
    h = int(hours)
    minutes = int((hours - h) * 60)
    seconds = (hours - h) * 3600 - minutes * 60
    sec = int(seconds)
    micros = int((seconds - sec) * 1e6)
    print(f"JD: {jd}, Date: {year}-{month:02d}-{day_int:02d} {h:02d}:{minutes:02d}:{sec:02d}.{micros:06d}")
    return datetime(year, month, day_int, h, minutes, sec, micros)

def julian_day_to_doy(julian_days):
    """
    Converts a Julian day number to a datetime object.

    Args:
        julian_day_number (float): The Julian day number.

    Returns:
        datetime.datetime: The corresponding datetime object.
    """
    output = []
    print()
    for julian_day_number in julian_days:
        # Calculate the difference in days from the reference point
        
        # Add the timedelta to the reference datetime
        result_datetime = julian_day_to_datetime(julian_day_number)
        output.append(result_datetime.timetuple().tm_yday)
        
    return output

def read_hsk_arcsix(hsk_filename):
    print('Reading {}'.format(hsk_filename))
    h5f = h5py.File(hsk_filename, 'r')
    # ARCSIX-HSK_P3B_20240725_v0.h5
    time_str = hsk_filename.split('_')[2]
    dt = datetime.strptime(time_str, '%Y%m%d')
    doy = dt.timetuple().tm_yday
    doy_repeat = np.repeat(doy, len(h5f['tmhr'][...]))
    
    # doys = julian_day_to_doy(h5f['jday'][...])
    # print('Date:', dt.strftime('%Y-%m-%d'), 'Doy:', doy)
    # sys.exit()
    hsk_data = {'doy': doy_repeat,
                'hrs': h5f['tmhr'][...]        ,
                'alt': h5f['alt'][...],
                'lat': h5f['lat'][...]    ,
                'lon': h5f['lon'][...]   ,
                'hed': h5f['ang_hed'][...],
                'rol': h5f['ang_rol'][...]  ,
                'pit': h5f['ang_pit'][...]  ,
                'sza': h5f['sza'][...],
                'saa': h5f['saa'][...],}
    return hsk_data

plot_dates = ['2024-05-28', '2024-05-30', '2024-05-31', 
              '2024-06-03', '2024-06-05', '2024-06-06',
              '2024-06-07', '2024-06-10', '2024-06-11', '2024-06-13', 
              '2024-07-25', '2024-07-29', '2024-07-30',
              '2024-08-01', '2024-08-02', '2024-08-07',
              '2024-08-08', '2024-08-09', '2024-08-15']

plot_dates_no_hyphen = [date.replace('-', '') for date in plot_dates]

flt_numbers = np.arange(1, len(plot_dates)+1)

flt_num_dict = {date: flt_num for date, flt_num in zip(plot_dates, flt_numbers)}

if __name__ == "__main__":
    # Specify multiple date ranges as a list of (start_date, end_date) tuples
    date_ranges = [
        ('2024-05-28', '2024-06-13'), # spring
        ('2024-07-25', '2024-08-15'), # summer
    ]

    cases = {
        '2024-05-31': [
            ['14:06:00', '14:16:12'],
            ['16:29:24', '16:43:12'],
        ],
        '2024-06-03': [
            ['14:43:12', '14:51:24'],
            ['14:57:00', '15:05:24'],      
        ],
        '2024-06-05': [
            ['15:33:00', '15:55:45'],
            ['16:02:35', '16:19:02'],      
        ],
        '2024-06-06': [
            ['13:59:24', '14:10:48'],
            ['14:15:24', '14:27:24'],
            ['16:32:24', '16:37:12'],
            ['16:51:00', '16:56:24'],
        ],
        '2024-06-07': [
            ['15:20:24', '15:45:30'],
            ['15:50:24', '16:15:55'],
        ],
        '2024-06-11': [
            ['13:54:25', '14:20:30'],
            ['15:21:10', '15:42:50'],
        ],
        '2024-06-13': [
            ['15:50:24', '15:52:48'],
            ['15:56:24', '15:58:48'],
            ['16:46:48', '16:51:00'],
            ['16:54:24', '17:00:00'],
            
        ],
        
    }
    
    clear_cases = {
        '2024-05-31': [
            ['14:06:00', '14:16:12'],
            ['16:29:24', '16:43:12'],
        ],
        '2024-06-05': [
            ['15:33:00', '15:55:45'],
            ['16:02:35', '16:19:02'],      
        ],
        '2024-06-06': [
            ['16:32:24', '16:37:12'],
            ['16:51:00', '16:56:24'],
        ],
        '2024-06-13': [
            ['16:46:48', '16:51:00'],
            ['16:54:24', '17:00:00'],
        ],
        
    }
    
    cloudy_cases = {
        '2024-06-03': [
            ['14:43:12', '14:51:24'],
            ['14:57:00', '15:05:24'],      
        ],
        '2024-06-06': [
            ['13:59:24', '14:10:48'],
            ['14:15:24', '14:27:24'],
        ],
        '2024-06-07': [
            ['15:20:24', '15:45:30'],
            ['15:50:24', '16:15:55'],
        ],
        '2024-06-13': [
            ['15:50:24', '15:52:48'],
            ['15:56:24', '15:58:48'],
        ],
        
    }

    # Specify the directory where HSK files are located
    hsk_dir = '../data/processed'
    hsk_files = glob.glob(os.path.join(hsk_dir, 'ARCSIX-HSK_P3B_*v0.h5'))
    print("Found HSK files:", hsk_files)

    # Extract the HSK data
    hsk_data_all = {}
    hsk_data_selected = {}
    lon_all = []
    lat_all = []
    for hsk_filename in hsk_files:
        file_date = datetime.strptime(os.path.basename(hsk_filename).split('_')[2], "%Y%m%d")
        in_range = False
        for start, end in date_ranges:
            start_dt = datetime.strptime(start, "%Y-%m-%d")
            end_dt = datetime.strptime(end, "%Y-%m-%d")
            if start_dt <= file_date <= end_dt:
                in_range = True
                break
        if not in_range:
            continue

        hsk_data = read_hsk_arcsix(hsk_filename)

        date_str = os.path.basename(hsk_filename).split('_')[2]

        if date_str in plot_dates_no_hyphen:
            hsk_data_all[date_str] = hsk_data # all dates
            lon_all.extend(hsk_data['lon'][...])
            lat_all.extend(hsk_data['lat'][...])


    # print("hsk_data_selected.items():", hsk_data_selected.items())
    # sys.exit()
    color_list = plt.get_cmap('jet_r', len(hsk_data_all))
    
    ### Plot

    cartopy_proj = ccrs.Orthographic(central_longitude=-45, central_latitude=80)

    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': cartopy_proj})

    # Set the extent for the main axes
    lon_min = np.min(lon_all)
    lon_max = np.max(lon_all)
    lat_min = np.min(lat_all)
    lat_max = np.max(lat_all)
    # expand the extent a bit
    lon_buffer = (lon_max - lon_min) * 0.0
    lat_buffer = (lat_max - lat_min) * 0.0
    lon_min -= lon_buffer
    lon_max += lon_buffer
    lat_min -= lat_buffer
    lat_max += lat_buffer
    ax.set_extent([lon_min, lon_max, lat_min, lat_max],
                  crs=ccrs.PlateCarree())

    # features
    ax.coastlines(linewidth=0.5, color='black')
    ax.add_feature(
        cfeature.LAND.with_scale('50m'),
        facecolor='white'
    )
    # ocean_color = '#f5fcff'
    ocean_color = '#9ce0ff'
    ax.add_feature(
        cfeature.OCEAN.with_scale('50m'),
        facecolor=ocean_color,
    )

    # Gridlines
    g1 = ax.gridlines(lw=0.5, color='gray', draw_labels=True, ls='--')
    g1.xlocator = FixedLocator(np.arange(-180, 180.1, 15.0))
    g1.ylocator = FixedLocator(np.arange(50, 90.1, 5.0))
    g1.top_labels = False

    # Plot all flight paths
    for idate, (date_str, data) in enumerate(hsk_data_all.items()):
        lon = data['lon'][...][::5]
        lat = data['lat'][...][::5]

        date_str_formatted = plot_dates[idate]
        
        color = color_list(idate)
        label = f'SF #{flt_num_dict[date_str_formatted]} on {date_str_formatted}'
        zorder = 6
        ax.plot(lon, lat, color=color, linewidth=1.5, transform=ccrs.PlateCarree(), label=label, zorder=zorder)

    
    
    leg = ax.legend(loc='center left', fontsize=9, bbox_to_anchor=(1.07, 0.5))
    leg.get_frame().set_alpha(0.925)
    leg.get_frame().set_facecolor('white')

    plt.savefig('arcsix_flight_paths_all.png', dpi=300, bbox_inches='tight')

    # plt.show()