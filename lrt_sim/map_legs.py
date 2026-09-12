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
    hsk_dir = 'data/processed'
    hsk_files = glob.glob(os.path.join(hsk_dir, 'ARCSIX-HSK_P3B_*.h5'))
    print("Found HSK files:", hsk_files)

    # Extract the HSK data
    hsk_data_all = {}
    hsk_data_selected = {}

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

        hsk_data_all[date_str] = hsk_data # all dates

        date_str_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        if date_str_formatted in cases: # specially selected dates/cases
            hsk_data_item = {}
            for time_range in cases[date_str_formatted]:
                start_time, end_time = time_range
                tmhr = hsk_data['hrs']
                start_hour, start_minute, start_second = map(int, start_time.split(':'))
                end_hour, end_minute, end_second = map(int, end_time.split(':'))
                start_decimal = start_hour + start_minute / 60. + start_second / 3600.
                end_decimal = end_hour + end_minute / 60. + end_second / 3600.
                time_mask = (tmhr >= start_decimal) & (tmhr <= end_decimal)
                selected_data = {k: v[time_mask] for k, v in hsk_data.items()}
                hsk_data_item[(start_time, end_time)] = selected_data
            hsk_data_selected[date_str_formatted] = hsk_data_item
    # print("hsk_data_selected.items():", hsk_data_selected.items())
    # sys.exit()
    color_list = plt.get_cmap('jet_r', len(hsk_data_selected))
    
    ### Plot

    cartopy_proj = ccrs.Orthographic(central_longitude=-45, central_latitude=80)

    fig, ax = plt.subplots(figsize=(6, 5), subplot_kw={'projection': cartopy_proj})

    # Set the extent for the main axes
    ax.set_extent([-85, -20, 74, 86], crs=ccrs.PlateCarree())

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

        if idate == 0:
            label = 'All flights'
        else:
            label = None
        
        date_str_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        if 0:#date_str_formatted in hsk_data_selected:
            color_idx = sorted(hsk_data_selected.keys()).index(date_str_formatted)
            color = color_list(color_idx)
            zorder = 6
        else:
            color = 'lightgrey'
            zorder = 1
        ax.plot(lon, lat, color=color, linewidth=0.5, transform=ccrs.PlateCarree(), label=label, zorder=zorder)

    # Plot selected flight paths
    all_sel_lons = []
    all_sel_lats = []
    # Sort by the date_str key for consistent color/label order
    for idate, (date_str, data) in enumerate(sorted(hsk_data_selected.items())):
        for itime, (time_range, selected_data) in enumerate(data.items()):
            lon = selected_data['lon'][...][::5]
            lat = selected_data['lat'][...][::5]
            all_sel_lons.extend(lon)
            all_sel_lats.extend(lat)
            if itime == 0:
                ax.plot(
                    lon, lat,
                    color=color_list(idate),
                    linewidth=3,
                    transform=ccrs.PlateCarree(),
                    label=f"Case {idate+1} ({date_str[5:7]}/{date_str[8:10]})"
                )
            else:
                ax.plot(lon, lat, color=color_list(idate), linewidth=2.5, transform=ccrs.PlateCarree())

    # # Inset axes for zoomed view
    if all_sel_lons and all_sel_lats:
        min_lon, max_lon = np.min(all_sel_lons), np.max(all_sel_lons)
        min_lat, max_lat = np.min(all_sel_lats), np.max(all_sel_lats)
        lon_margin = (max_lon - min_lon) * 0.05
        lat_margin = (max_lat - min_lat) * 0.05
        min_lon -= lon_margin
        max_lon += lon_margin
        min_lat -= lat_margin
        max_lat += lat_margin

        axins = ax.inset_axes([0.45, 0.01, 0.54, 0.54], transform=ax.transAxes, projection=cartopy_proj)
        axins.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
        axins.coastlines(linewidth=0.5, color='black')
        axins.add_feature(cfeature.LAND.with_scale('50m'), facecolor='white')
        axins.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor=ocean_color)

        for date_str, data in hsk_data_all.items():
            lon = data['lon'][...][::5]
            lat = data['lat'][...][::5]

            date_str_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
            if 0:# date_str_formatted in hsk_data_selected:
                color_idx = sorted(hsk_data_selected.keys()).index(date_str_formatted)
                color = color_list(color_idx)
                zorder = 6
            else:
                color = 'lightgrey'
                zorder = 1
            axins.plot(lon, lat, color=color, linewidth=0.5, transform=ccrs.PlateCarree(), label=label, zorder=zorder)

        for idate, (date_str, data) in enumerate(sorted(hsk_data_selected.items())):
            for itime, (time_range, selected_data) in enumerate(data.items()):
                lon = selected_data['lon'][...][::5]
                lat = selected_data['lat'][...][::5]
                axins.plot(lon, lat, color=color_list(idate), linewidth=4, transform=ccrs.PlateCarree())

        axins.set_xticks([])
        axins.set_yticks([])

        # Make the outer border of the inset axes thicker
        for spine in axins.spines.values():
            spine.set_linewidth(1.5)

        # Get the corners of the inset axes in axes fraction coordinates
        corners_axes = np.array([
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 0],
            [0, 0]  # Closing the loop
        ])
        # Convert these to display coordinates, then to data coordinates in PlateCarree
        corner_lons = []
        corner_lats = []
        for xy in corners_axes:
            disp = axins.transAxes.transform(xy)
            data = axins.transData.inverted().transform(disp)
            lonlat = ccrs.PlateCarree().transform_point(
                data[0], data[1], axins.projection
            )
            corner_lons.append(lonlat[0])
            corner_lats.append(lonlat[1])

        print("Inset axes corners (lon, lat):")
        for i in range(4):
            print(f"  {corners_axes[i]}: ({corner_lons[i]:.6f}, {corner_lats[i]:.6f})")

        # Add a box to the main axes to indicate the inset area
        ax.plot(
            corner_lons[:], corner_lats[:],
            color='black', linewidth=1., linestyle='solid',
            transform=ccrs.PlateCarree(), zorder=10
        )
    
    leg = ax.legend(loc='upper left', fontsize=9)
    leg.get_frame().set_alpha(0.925)
    leg.get_frame().set_facecolor('white')

    plt.savefig('arcsix_flight_paths.png', dpi=300, bbox_inches='tight')

    # plt.show()