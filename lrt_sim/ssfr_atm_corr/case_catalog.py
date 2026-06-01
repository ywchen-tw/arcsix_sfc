"""Active cases from the original SSFR atmospheric-correction script.

This catalog keeps the cases from ``legacy/ssfr_atm_corr_ori.py`` after line
2100. Cases with custom atmospheric grids store the recovered grid directly in
``levels``; default-grid cases use ``levels=None``.
"""

import datetime
import csv
import filecmp
import glob
import math
import os
import re
import numpy as np

if __package__:
    from .settings import _fdir_general_, _mission_, _platform_
else:
    from settings import _fdir_general_, _mission_, _platform_


DEFAULT_CLOSURE_THRESHOLDS = {
    'fdn_abs_broadband_bias': 0.05,
    'fdn_flux_weighted_relative_rmse': 0.05,
    'fup_abs_broadband_bias': 0.05,
    'fup_flux_weighted_relative_rmse': 0.08,
}


def cloud_observation_file(fdir_general, mission, platform_name, date_s, case_tag, time_start, time_end):
    """Return the expected preprocessed cloud-observation file for one time window."""
    return (
        '%s/flt_cld_obs_info/%s_cld_obs_info_%s_%s_%s_time_%.3f-%.3f_atm_corr.pkl'
        % (
            fdir_general,
            mission.lower(),
            platform_name.lower(),
            date_s,
            case_tag,
            time_start,
            time_end,
        )
    )


def split_case_tmhr_ranges(tmhr_ranges_select, simulation_interval):
    """Split selected time ranges using the same rule as the workflow setup."""
    if simulation_interval is None:
        return tmhr_ranges_select

    split_ranges = []
    for lo, hi in tmhr_ranges_select:
        t_start = lo
        while t_start < hi and t_start < (hi - 0.0167 / 6):
            t_end = min(t_start + simulation_interval / 60.0, hi)
            split_ranges.append([t_start, t_end])
            t_start = t_end
    return split_ranges


def catalog_case_levels(case):
    """Return custom atmospheric levels for a catalog case, or None for default levels."""
    levels = case.get('levels')
    if levels is None:
        return None

    levels = np.asarray(levels, dtype=float)
    if levels.ndim != 1 or levels.size == 0:
        raise ValueError(f"{case['id']} custom levels must be a non-empty 1-D array.")
    return levels


def missing_cloud_observation_files(case, date_s):
    """Return missing preprocessed cloud-observation files needed by a catalog case."""
    tmhr_ranges_select = split_case_tmhr_ranges(
        case['tmhr_ranges_select'],
        case['simulation_interval'],
    )
    expected_files = [
        cloud_observation_file(
            _fdir_general_,
            _mission_,
            _platform_,
            date_s,
            case['case_tag'],
            time_start,
            time_end,
        )
        for time_start, time_end in tmhr_ranges_select
    ]
    return [fname for fname in expected_files if not os.path.exists(fname)]


def closure_metric_status(output_file, thresholds):
    """Return closure-check details for one simulation CSV."""
    try:
        metrics = closure_metrics(output_file)
    except ValueError as err:
        return False, [str(err)]

    checks = [(name, metrics[name], thresholds[name]) for name in metrics]
    failed_checks = [
        f'{name}={value:.5f} > {threshold:.5f}'
        for name, value, threshold in checks
        if value > threshold
    ]
    return len(failed_checks) == 0, failed_checks


def closure_metrics(output_file):
    """Return the closure metrics stored in one simulation CSV."""
    required_columns = [
        'fdn_broadband_bias',
        'fdn_flux_weighted_relative_rmse',
        'fup_broadband_bias',
        'fup_flux_weighted_relative_rmse',
    ]
    with open(output_file, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f'{output_file} has no data rows')

    row = rows[0]
    missing_columns = [column for column in required_columns if column not in row]
    if missing_columns:
        return reconstructed_closure_metrics(output_file, rows)

    metrics = {
        'fdn_abs_broadband_bias': abs(float(row['fdn_broadband_bias'])),
        'fdn_flux_weighted_relative_rmse': float(row['fdn_flux_weighted_relative_rmse']),
        'fup_abs_broadband_bias': abs(float(row['fup_broadband_bias'])),
        'fup_flux_weighted_relative_rmse': float(row['fup_flux_weighted_relative_rmse']),
    }
    invalid_metrics = [name for name, value in metrics.items() if not math.isfinite(value)]
    if invalid_metrics:
        raise ValueError(f'{output_file} has non-finite metrics: {", ".join(invalid_metrics)}')

    return metrics


def finite_csv_float(value):
    """Return a finite float from a CSV value, or None."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def pair_closure_metrics(rows, output_file, obs_column, sim_column):
    """Calculate broadband bias and flux-weighted relative RMSE from spectral rows."""
    missing_columns = [
        column for column in (obs_column, sim_column)
        if column not in rows[0]
    ]
    if missing_columns:
        raise ValueError(f'{output_file} missing columns: {", ".join(missing_columns)}')

    pairs = []
    for row in rows:
        obs = finite_csv_float(row.get(obs_column))
        sim = finite_csv_float(row.get(sim_column))
        if obs is not None and sim is not None:
            pairs.append((obs, sim))

    if not pairs:
        raise ValueError(f'{output_file} has no finite {obs_column}/{sim_column} pairs')

    obs_sum = sum(obs for obs, _ in pairs)
    if obs_sum == 0:
        raise ValueError(f'{output_file} has zero {obs_column} sum')

    broadband_bias = sum(sim - obs for obs, sim in pairs) / obs_sum
    weighted_pairs = [(obs, sim) for obs, sim in pairs if obs > 0]
    obs_weighted_sum = sum(obs for obs, _ in weighted_pairs)
    if obs_weighted_sum == 0:
        raise ValueError(f'{output_file} has no positive {obs_column} values for weighted RMSE')

    flux_weighted_relative_rmse = math.sqrt(
        sum(
            (obs / obs_weighted_sum) * ((sim - obs) / obs)**2
            for obs, sim in weighted_pairs
        )
    )
    return broadband_bias, flux_weighted_relative_rmse


def reconstructed_closure_metrics(output_file, rows):
    """Reconstruct closure metrics from older simulation CSVs."""
    fdn_broadband_bias, fdn_flux_weighted_relative_rmse = pair_closure_metrics(
        rows,
        output_file,
        'ssfr_fdn_mean',
        'simu_fdn_mean',
    )
    fup_broadband_bias, fup_flux_weighted_relative_rmse = pair_closure_metrics(
        rows,
        output_file,
        'ssfr_fup_mean',
        'simu_fup_mean',
    )
    return {
        'fdn_abs_broadband_bias': abs(fdn_broadband_bias),
        'fdn_flux_weighted_relative_rmse': fdn_flux_weighted_relative_rmse,
        'fup_abs_broadband_bias': abs(fup_broadband_bias),
        'fup_flux_weighted_relative_rmse': fup_flux_weighted_relative_rmse,
    }


def mean_closure_metrics(output_files):
    """Return mean closure metrics across all simulation CSV files."""
    metrics_by_file = [closure_metrics(output_file) for output_file in output_files]
    return {
        metric_name: sum(metrics[metric_name] for metrics in metrics_by_file) / len(metrics_by_file)
        for metric_name in metrics_by_file[0]
    }


def mean_closure_metric_status(output_files, thresholds):
    """Return whether mean closure metrics pass, plus failed mean checks."""
    metrics_by_file = []
    invalid_files = []
    for output_file in output_files:
        try:
            metrics_by_file.append(closure_metrics(output_file))
        except ValueError as err:
            invalid_files.append(str(err))

    if metrics_by_file:
        metrics = {
            metric_name: sum(metrics[metric_name] for metrics in metrics_by_file) / len(metrics_by_file)
            for metric_name in metrics_by_file[0]
        }
    else:
        metrics = {}

    failed_checks = [
        f'mean_{name}={value:.5f} > {thresholds[name]:.5f}'
        for name, value in metrics.items()
        if value > thresholds[name]
    ]
    failed_checks.extend(invalid_files)
    return len(failed_checks) == 0, failed_checks, metrics

ALL_CASE_CATALOG = [{'id': 'case_029',
  'date': '2024-05-28',
  'case_tag': 'clear_atm_corr',
  'tmhr_ranges_select': [
                        [15.610, 15.822],
                        [16.905, 17.404] 
                        ],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'levels': None,
  },
                    
 {'id': 'case_030',
  'date': '2024-05-31',
  'case_tag': 'clear_atm_corr',
  'tmhr_ranges_select': [
                        [13.839, 15.180],  # 5.6 km
                        ],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'levels': None,
  }
 ,
 {'id': 'case_031',
  'date': '2024-05-31',
  'case_tag': 'clear_atm_corr_2',
  'tmhr_ranges_select': [
                        [16.905, 17.404] 
                        ],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'levels': None,
  },
 
 {'id': 'case_032',
  'date': '2024-06-03',
  'case_tag': 'cloudy_atm_corr_1',
  'tmhr_ranges_select': [[13.62, 13.75],  # 300m, cloudy, camera icing
                                            ],
  'simulation_interval': 0.5,
  'clear_sky': False,
  'manual_cloud': True,
  'levels': np.concatenate((np.array([0.0, 0.2, 0.3, 0.4, 0.7, 1.0,]),
                            np.array([1.41, 1.5, 1.93, 2.0, 2.5, 3.0, 4.0]), 
                            np.arange(5.0, 10.1, 2.5),
                            np.array([15, 20, 30., 40., 45.]))),
  'manual_cloud_cer': 13.0,
  'manual_cloud_cwp': 0.07782,
  'manual_cloud_cwp_expr': '77.82 / 1000',
  'manual_cloud_cth': 1.93,
  'manual_cloud_cbh': 1.41,
  'manual_cloud_cot': 21.27,},
 
 {'id': 'case_033',
  'date': '2024-06-03',
  'case_tag': 'cloudy_atm_corr_2',
  'tmhr_ranges_select': [
                        [14.711, 14.868],  # 300m, cloudy, camera icing
                        ],
  'simulation_interval': 0.5,
  'clear_sky': False,
  'manual_cloud': True,
  'levels': np.concatenate((np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0,]),
                            np.array([1.5, 1.91, 2.0, 2.5, 3.0, 4.0]), 
                            np.arange(5.0, 10.1, 2.5),
                            np.array([15, 20, 30., 40., 45.]))),
  'manual_cloud_cer': 7.0,
  'manual_cloud_cwp': 0.11365,
  'manual_cloud_cwp_expr': '113.65 / 1000',
  'manual_cloud_cth': 1.91,
  'manual_cloud_cbh': 0.5,
  'manual_cloud_cot': 24.31,},
 
 {'id': 'case_034',
  'date': '2024-06-05',
  'case_tag': 'clear_atm_corr_1',
  'tmhr_ranges_select': [
                        [12.405, 13.812], # 5.7m,
                        ],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'levels': None,
  },
 
 {'id': 'case_035',
  'date': '2024-06-05',
  'case_tag': 'clear_atm_corr_2',
  'tmhr_ranges_select': [
                        [14.258, 15.036], # 100m
                        ],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'levels': None,
  },
 
 {'id': 'case_036',
  'date': '2024-06-05',
  'case_tag': 'clear_atm_corr_3',
  'tmhr_ranges_select': [
                        [15.535, 15.931], # 450m
                        ],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'levels': None,
  },
 
 {'id': 'case_037',
  'date': '2024-06-05',
  'case_tag': 'clear_sky_spiral_atm_corr',
  'tmhr_ranges_select': [
                        [13.7889, 13.8010],
                        [13.8350, 13.8395],
                        [13.8780, 13.8885],
                        [13.9240, 13.9255],
                        [13.9389, 13.9403],
                        [13.9540, 13.9715],
                        [13.9980, 14.0153],
                        [14.0417, 14.0575],
                        [14.0417, 14.0475],
                        [14.0560, 14.0590],
                        [14.0825, 14.0975],
                        [14.1264, 14.1525],
                        [14.1762, 14.1975],
                        [14.2194, 14.2420],
                        [14.2605, 14.2810]
                        ],
  'simulation_interval': None,
  'clear_sky': True,
  'levels': None,
  },
 
 {'id': 'case_038',
  'date': '2024-06-06',
  'case_tag': 'clear_atm_corr',
  'tmhr_ranges_select': [
                        [16.250, 16.325], # 100m, 
                        [16.375, 16.632], # 450m
                        [16.700, 16.794], # 100m
                        [16.850, 16.952], # 1.2km
                        ],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'levels': None,
  },
 
 {'id': 'case_039',
  'date': '2024-06-07',
  'case_tag': 'cloudy_atm_corr',
  'tmhr_ranges_select': [
                        [15.319, 15.763], # 100m, cloudy
                        ],
  'simulation_interval': 0.5,
  'clear_sky': False,
  'manual_cloud': True,
  'levels': np.concatenate((np.array([0.0, 0.1, 0.15, 0.2, 0.43, 0.5, 0.6, 0.8, 1.0,]),
                            np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
                            np.arange(5.0, 10.1, 2.5),
                            np.array([15, 20, 30., 40., 45.]))),
  'manual_cloud_cer': 6.7,
  'manual_cloud_cwp': 0.02696,
  'manual_cloud_cwp_expr': '26.96 / 1000',
  'manual_cloud_cth': 0.43,
  'manual_cloud_cbh': 0.15,
  'manual_cloud_cot': 6.02,},
 
 {'id': 'case_040',
  'date': '2024-06-11',
  'case_tag': 'clear_sky_spiral_atm_corr',
  'tmhr_ranges_select': [
                        [14.5667, 14.5694],
                        [14.5986, 14.6097],
                        [14.6375, 14.6486], # cloud shadow
                        [14.6778, 14.6903],
                        [14.7208, 14.7403],
                        [14.7653, 14.7875],
                        [14.8125, 14.8278],
                        [14.8542, 14.8736],
                        [14.8986, 14.9389], # more cracks
                        ],
  'simulation_interval': None,
  'clear_sky': True,
  'levels': None,
  },
 
 {'id': 'case_041',
  'date': '2024-06-11',
  'case_tag': 'clear_atm_corr_1',
  'tmhr_ranges_select': [
                        [14.968, 15.229], # 100, clear, some cloud
                        [14.968, 15.347],
                        ],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'levels': None,
  },
 
 {'id': 'case_042',
  'date': '2024-06-11',
  'case_tag': 'clear_atm_corr_2',
  'tmhr_ranges_select': [
                        [15.347, 15.813], # 100m
                        [15.813, 16.115], # 100-450m, clear, some cloud
                        ],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'levels': None,
  },
 
 {'id': 'case_043',
  'date': '2024-06-13',
  'case_tag': 'clear_atm_corr_1',
  'tmhr_ranges_select': [[13.704, 13.817], # 100-450m, clear, some cloud
                                            ],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'levels': None,
  },
 
 {'id': 'case_044',
  'date': '2024-06-13',
  'case_tag': 'cloudy_atm_corr_1',
  'tmhr_ranges_select': [
                        [14.109, 14.140], # 100m, cloudy
                        ],
  'simulation_interval': 0.5,
  'clear_sky': False,
  'manual_cloud': True,
  'levels': np.concatenate((np.array([0.0, 0.1, 0.15, 0.2, 0.4, 0.52, 0.6, 0.8, 1.0,]),
                            np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
                            np.arange(5.0, 10.1, 2.5),
                            np.array([15, 20, 30., 40., 45.]))),
  'manual_cloud_cer': 17.4,
  'manual_cloud_cwp': 0.09051,
  'manual_cloud_cwp_expr': '90.51 / 1000',
  'manual_cloud_cth': 0.52,
  'manual_cloud_cbh': 0.15,
  'manual_cloud_cot': 7.82,},
 
 {'id': 'case_045',
  'date': '2024-06-13',
  'case_tag': 'cloudy_atm_corr_2',
  'tmhr_ranges_select': [
                        [15.834, 15.883], # 100m, cloudy
                        ],
  'simulation_interval': 0.5,
  'clear_sky': False,
  'manual_cloud': True,
  'levels': np.concatenate((np.array([0.0, 0.1, 0.2, 0.28, 0.3, 0.5, 0.58, 0.8, 1.0,]),
                            np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
                            np.arange(5.0, 10.1, 2.5),
                            np.array([15, 20, 30., 40., 45.]))),
  'manual_cloud_cer': 22.4,
  'manual_cloud_cwp': 0.0356,
  'manual_cloud_cwp_expr': '35.6 / 1000',
  'manual_cloud_cth': 0.58,
  'manual_cloud_cbh': 0.28,
  'manual_cloud_cot': 2.39,},
 
 {'id': 'case_046',
  'date': '2024-06-13',
  'case_tag': 'cloudy_atm_corr_3',
  'tmhr_ranges_select': [
                        [16.043, 16.067], # 100-200m, cloudy
                        ],
  'simulation_interval': 0.5,
  'clear_sky': False,
  'manual_cloud': True,
  'levels': np.concatenate((np.array([0.0, 0.1, 0.2, 0.3, 0.38, 0.5, 0.68, 0.8, 1.0,]),
                            np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
                            np.arange(5.0, 10.1, 2.5),
                            np.array([15, 20, 30., 40., 45.]))),
  'manual_cloud_cer': 8.9,
  'manual_cloud_cwp': 0.02129,
  'manual_cloud_cwp_expr': '21.29 / 1000',
  'manual_cloud_cth': 0.68,
  'manual_cloud_cbh': 0.38,
  'manual_cloud_cot': 3.59,},
 
 {'id': 'case_047',
  'date': '2024-06-13',
  'case_tag': 'clear_atm_corr_2',
  'tmhr_ranges_select': [
                        [16.550, 17.581], # 100-500m, clear
                        ],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'levels': None,
  },
 
 {'id': 'case_048',
  'date': '2024-07-25',
  'case_tag': 'cloudy_atm_corr',
  'tmhr_ranges_select': [
                        [15.094, 15.300], # 100m, some low clouds or fog below
                        ],
  'simulation_interval': 0.5,
  'clear_sky': False,
  'manual_cloud': True,
  'levels': np.concatenate((np.array([0.0, 0.1, 0.16, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0,]),
                            np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
                            np.arange(5.0, 10.1, 2.5),
                            np.array([15, 20, 30., 40., 45.]))),
  'manual_cloud_cer': 11.4,
  'manual_cloud_cwp': 0.00994,
  'manual_cloud_cwp_expr': '9.94 / 1000',
  'manual_cloud_cth': 0.3,
  'manual_cloud_cbh': 0.16,
  'manual_cloud_cot': 1.31,},
 
 {'id': 'case_049',
  'date': '2024-07-25',
  'case_tag': 'cloudy_atm_corr_2',
  'tmhr_ranges_select': [
                        [15.881, 15.903], # 200-500m
                        ],
  'simulation_interval': 0.5,
  'clear_sky': False,
  'manual_cloud': True,
  'levels': np.concatenate((np.array([0.0, 0.1, 0.16, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0,]),
                            np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
                            np.arange(5.0, 10.1, 2.5),
                            np.array([15, 20, 30., 40., 45.]))),
  'manual_cloud_cer': 11.4,
  'manual_cloud_cwp': 0.00994,
  'manual_cloud_cwp_expr': '9.94 / 1000',
  'manual_cloud_cth': 0.3,
  'manual_cloud_cbh': 0.16,
  'manual_cloud_cot': 1.31,},
 
 {'id': 'case_050',
  'date': '2024-07-29',
  'case_tag': 'clear_sky_spiral_atm_corr',
  'tmhr_ranges_select': [
                        [13.442, 13.465],
                        [13.490, 13.514],
                        [13.536, 13.554],
                        [13.580, 13.611],
                        [13.639, 13.654],
                        [13.676, 13.707],
                        [13.733, 13.775],
                        [13.793, 13.836],
                        ],
  'simulation_interval': None,
  'clear_sky': True,
  'levels': None,
  },
 
 {'id': 'case_051',
  'date': '2024-07-29',
  'case_tag': 'clear_atm_corr_1',
  'tmhr_ranges_select': [
                        [13.939, 14.200], # 100m, clear
                        [14.438, 14.714], # 3.7km
                        ],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'levels': None,
  },
 
 {'id': 'case_052',
  'date': '2024-07-29',
  'case_tag': 'clear_atm_corr_2',
  'tmhr_ranges_select': [
                        [15.214, 15.804], # 1.3km
                        [16.176, 16.304], # 1.3km
                        ],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'levels': None,
  },
 
 {'id': 'case_053',
  'date': '2024-07-30',
  'case_tag': 'clear_sky_spiral_atm_corr',
  'tmhr_ranges_select': [
                        [13.886, 13.908],
                        [13.934, 13.950],
                        [13.976, 14.000],
                        [14.031, 14.051],
                        [14.073, 14.096],
                        [14.115, 14.134],
                        [14.157, 14.179],
                        [14.202, 14.219],
                        [14.239, 14.254],
                        [14.275, 14.294],
                        ],
  'simulation_interval': None,
  'clear_sky': True,
  'levels': None,
  },
 
 {'id': 'case_054',
  'date': '2024-07-30',
  'case_tag': 'clear_atm_corr',
  'tmhr_ranges_select': [
                        [14.318, 14.936], # 100-450m, clear
                        [15.043, 15.140], # 1.5km
                        ],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'levels': None,
  },
 
 {'id': 'case_055',
  'date': '2024-08-01',
  'case_tag': 'clear_atm_corr',
  'tmhr_ranges_select': [
                        [13.843, 14.361], # 100-450m, clear, some open ocean
                        ],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'levels': None,
  },
 
 {'id': 'case_056',
  'date': '2024-08-01',
  'case_tag': 'clear_atm_corr_2',
  'tmhr_ranges_select': [
                        [14.739, 15.053], # 550m
                        ],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'levels': None,
  },
 
 {'id': 'case_057',
  'date': '2024-08-02',
  'case_tag': 'clear_atm_corr_1',
  'tmhr_ranges_select': [
                        [14.557, 15.100], # 100m
                        ],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'levels': None,
  },
 
 {'id': 'case_058',
  'date': '2024-08-02',
  'case_tag': 'clear_atm_corr_2',
  'tmhr_ranges_select': [
                        [15.244, 16.635], # 1km
                        ],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'levels': None,
  },
 
 {'id': 'case_059',
  'date': '2024-08-07',
  'case_tag': 'clear_atm_corr_1',
  'tmhr_ranges_select': [[13.344, 13.763], # 100m, cloudy
                                            ],
  'simulation_interval': 0.5,
  'clear_sky': False,
  'manual_cloud': True,
  'levels': np.concatenate((np.array([0.0, 0.1, 0.2, 0.4, 0.6, 0.65, 0.69, 0.78, 1.0,]),
                            np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
                            np.arange(5.0, 10.1, 2.5),
                            np.array([15, 20, 30., 40., 45.]))),
  'manual_cloud_cer': 10.7,
  'manual_cloud_cwp': 0.01128,
  'manual_cloud_cwp_expr': '11.28 / 1000',
  'manual_cloud_cth': 0.78,
  'manual_cloud_cbh': 0.69,
  'manual_cloud_cot': 1.59,},
 
 {'id': 'case_060',
  'date': '2024-08-07',
  'case_tag': 'cloudy_atm_corr_2',
  'tmhr_ranges_select': [
                        [15.472, 15.567], # 180m, cloudy
                        [15.580, 15.921], # 100m, cloudy
                        ],
  'simulation_interval': 0.5,
  'clear_sky': False,
  'manual_cloud': True,
  'levels': np.concatenate((np.array([0.0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.62, 0.8, 0.96,]),
                            np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
                            np.arange(5.0, 10.1, 2.5),
                            np.array([15, 20, 30., 40., 45.]))),
  'manual_cloud_cer': 7.2,
  'manual_cloud_cwp': 0.0775,
  'manual_cloud_cwp_expr': '77.5 / 1000',
  'manual_cloud_cth': 0.96,
  'manual_cloud_cbh': 0.62,
  'manual_cloud_cot': 16.21,},
 
 {'id': 'case_061',
  'date': '2024-08-08',
  'case_tag': 'clear_atm_corr_1',
  'tmhr_ranges_select': [
                        [12.990, 13.180], # 180m, clear
                        ],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'levels': None,
  },
 
 {'id': 'case_062',
  'date': '2024-08-08',
  'case_tag': 'clear_atm_corr_2',
  'tmhr_ranges_select': [
                        [14.250, 14.373], # 180m, clear
                        ],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'levels': None,
  },
 
 {'id': 'case_063',
  'date': '2024-08-08',
  'case_tag': 'clear_atm_corr_3',
  'tmhr_ranges_select': [
                        [16.471, 16.601], # 180m, clear
                        ],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'levels': None,
  },
 
 {'id': 'case_064',
  'date': '2024-08-08',
  'case_tag': 'cloudy_atm_corr_1',
  'tmhr_ranges_select': [
                        [13.212, 13.347], # 100m, cloudy
                        ],
  'simulation_interval': 0.5,
  'clear_sky': False,
  'manual_cloud': True,
  'levels': np.concatenate((np.array([0.0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.67, 0.8, 1.0,]),
                            np.array([1.5, 1.98, 2.5, 3.0, 4.0]), 
                            np.arange(5.0, 10.1, 2.5),
                            np.array([15, 20, 30., 40., 45.]))),
  'manual_cloud_cer': 15.3,
  'manual_cloud_cwp': 0.14394,
  'manual_cloud_cwp_expr': '143.94 / 1000',
  'manual_cloud_cth': 1.98,
  'manual_cloud_cbh': 0.67,
  'manual_cloud_cot': 14.12,},
 
 {'id': 'case_065',
  'date': '2024-08-08',
  'case_tag': 'cloudy_atm_corr_2',
  'tmhr_ranges_select': [
                        [15.314, 15.504], # 100m, cloudy
                        ],
  'simulation_interval': 0.5,
  'clear_sky': False,
  'manual_cloud': True,
  'levels': np.concatenate((np.array([0.0, 0.1, 0.2, 0.4, 0.78, 1.0,]),
                            np.array([1.5, 1.81, 2.21, 2.5, 3.0, 4.0]), 
                            np.arange(5.0, 10.1, 2.5),
                            np.array([15, 20, 30., 40., 45.]))),
  'manual_cloud_cer': 7.8,
  'manual_cloud_cwp': 0.06418,
  'manual_cloud_cwp_expr': '64.18 / 1000',
  'manual_cloud_cth': 2.21,
  'manual_cloud_cbh': 1.81,
  'manual_cloud_cot': 12.41,},
 
 {'id': 'case_066',
  'date': '2024-08-09',
  'case_tag': 'cloudy_atm_corr_1',
  'tmhr_ranges_select': [
                        [13.376, 13.600], # 100m, cloudy
                        ],
  'simulation_interval': 0.5,
  'clear_sky': False,
  'manual_cloud': True,
  'levels': np.concatenate((np.array([0.0, 0.1, 0.15, 0.2, 0.34, 0.4, 0.6, 0.77, 1.0,]),
                                               np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
                                               np.arange(5.0, 10.1, 2.5),
                                               np.array([15, 20, 30., 40., 45.]))),
  'manual_cloud_cer': 9.0,
  'manual_cloud_cwp': 0.08349,
  'manual_cloud_cwp_expr': '83.49 / 1000',
  'manual_cloud_cth': 0.77,
  'manual_cloud_cbh': 0.34,
  'manual_cloud_cot': 13.93,},
 
 {'id': 'case_067',
  'date': '2024-08-09',
  'case_tag': 'clear_atm_corr',
  'tmhr_ranges_select': [
                        [14.750, 15.060], # 100m, clear
                        [15.622, 15.887], # 100m, clear
                        ],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'levels': None,
  },
 
 {'id': 'case_068',
  'date': '2024-08-09',
  'case_tag': 'cloudy_atm_corr_2',
  'tmhr_ranges_select': [
                        [16.029, 16.224], # 100m, cloudy
                        ],
  'simulation_interval': 0.5,
  'clear_sky': False,
  'manual_cloud': True,
  'levels': np.concatenate((np.array([0.0, 0.1, 0.2, 0.29, 0.4, 0.62, 0.8, 1.0,]),
                                               np.array([1.5, 2.0, 2.5, 3.0, 4.0]), 
                                               np.arange(5.0, 10.1, 2.5),
                                               np.array([15, 20, 30., 40., 45.]))),
  'manual_cloud_cer': 8.3,
  'manual_cloud_cwp': 0.0491,
  'manual_cloud_cwp_expr': '49.1 / 1000',
  'manual_cloud_cth': 0.62,
  'manual_cloud_cbh': 0.29,
  'manual_cloud_cot': 8.93,},
 
 {'id': 'case_069',
  'date': '2024-08-15',
  'case_tag': 'clear_atm_corr',
  'tmhr_ranges_select': [
                        [14.085, 14.396], # 100m, clear
                        [14.550, 14.968], # 3.5km, clear
                        [15.078, 15.163], # 1.7km, clear
                        ],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'levels': None,
  }
]


# Active cases recovered from legacy/ssfr_atm_corr_ori.py after line 2100.
CASE_CATALOG = ALL_CASE_CATALOG


def cases_for_date(date_s):
    """Return flight-track catalog entries for a YYYYMMDD or YYYY-MM-DD date string."""
    normalized = date_s if '-' in date_s else f'{date_s[:4]}-{date_s[4:6]}-{date_s[6:8]}'
    return [case for case in CASE_CATALOG if case['date'] == normalized]


def get_case(case_id):
    """Return one flight-track catalog entry by id."""
    for case in CASE_CATALOG:
        if case['id'] == case_id:
            return case
    raise KeyError(case_id)


def iteration_closure_check(output_files, thresholds=None):
    """Return True when mean closure metrics meet flux-closure thresholds."""
    if thresholds is None:
        thresholds = DEFAULT_CLOSURE_THRESHOLDS
    if not output_files:
        return False

    passed, _, _ = mean_closure_metric_status(output_files, thresholds)
    return passed


def iteration_output_files(case, date_s, iter):
    """Return simulation CSV files generated for one catalog iteration."""
    sky_tag = 'clear' if case['clear_sky'] else 'sat_cloud'
    output_patterns = [
        os.path.join(
            _fdir_general_,
            'lrt',
            f"{date_s}_{case['case_tag']}_{sky_tag}",
            f'ssfr_simu_flux_{date_s}_*_iteration_{iter}.csv',
        ),
        os.path.join(
            _fdir_general_,
            f'ssfr_simu_flux_{date_s}_*_iteration_{iter}.csv',
        ),
    ]
    output_files = []
    for output_pattern in output_patterns:
        output_files.extend(glob.glob(output_pattern))
    return sorted(set(output_files))


def case_lrt_output_dir(case, date_s):
    """Return the libRadtran output directory for a catalog case."""
    sky_tag = 'clear' if case['clear_sky'] else 'sat_cloud'
    return os.path.join(_fdir_general_, 'lrt', f"{date_s}_{case['case_tag']}_{sky_tag}")


def final_native_output_files(case, date_s):
    """Return native-grid final CSV files expected for a completed case."""
    output_dir = case_lrt_output_dir(case, date_s)
    native_files = []
    missing_patterns = []
    tmhr_ranges_select = split_case_tmhr_ranges(
        case['tmhr_ranges_select'],
        case['simulation_interval'],
    )
    for time_start, time_end in tmhr_ranges_select:
        pattern = os.path.join(
            output_dir,
            f'ssfr_simu_flux_{date_s}_{time_start:.3f}-{time_end:.3f}_alt-*km_final.csv',
        )
        matches = sorted(glob.glob(pattern))
        if matches:
            native_files.extend(matches)
        else:
            missing_patterns.append(pattern)
    return sorted(set(native_files)), missing_patterns


def iteration_albedo_files(case, date_s, iter):
    """Return native-grid albedo files generated for one catalog iteration."""
    tmhr_ranges_select = split_case_tmhr_ranges(
        case['tmhr_ranges_select'],
        case['simulation_interval'],
    )
    pattern = os.path.join(
        _fdir_general_,
        'sfc_alb',
        f'sfc_alb_{date_s}_*km_iter_{iter}.dat',
    )
    candidates = sorted(glob.glob(pattern))

    albedo_files = []
    missing_ranges = []
    for time_start, time_end in tmhr_ranges_select:
        matches = [
            candidate
            for candidate in candidates
            if albedo_file_matches_time_range(candidate, time_start, time_end)
        ]
        if matches:
            albedo_files.extend(matches)
        else:
            missing_ranges.append(f'{time_start:.3f}-{time_end:.3f}')
    return sorted(set(albedo_files)), missing_ranges


def albedo_file_matches_time_range(albedo_file, time_start, time_end, tolerance=0.002):
    """Return True when an albedo filename time range belongs to a split leg."""
    match = re.search(
        r'sfc_alb_\d{8}_(\d+\.\d{3})_(\d+\.\d{3})_-?\d+\.\d+km_iter_\d+\.dat$',
        os.path.basename(albedo_file),
    )
    if match is None:
        return False

    file_start, file_end = [float(value) for value in match.groups()]
    file_midpoint = 0.5 * (file_start + file_end)
    return (
        file_start >= time_start - tolerance
        and file_end <= time_end + tolerance
        and time_start - tolerance <= file_midpoint <= time_end + tolerance
    )


def summarize_missing_ranges(missing_ranges, max_items=5):
    """Return a compact missing-range summary for logging."""
    if len(missing_ranges) <= max_items:
        return ', '.join(missing_ranges)
    shown_ranges = ', '.join(missing_ranges[:max_items])
    return f'{shown_ranges}, ...'


def missing_final_extension_outputs(native_final_files):
    """Return missing final-extension CSV/albedo files for native final files."""
    missing = []
    for native_final_file in native_final_files:
        final_extension_file = native_final_file.replace('_final.csv', '_final_extension.csv')
        if not os.path.exists(final_extension_file):
            missing.append(final_extension_file)
        albedo_extension_file = final_albedo_extension_file(native_final_file)
        if albedo_extension_file is not None and not os.path.exists(albedo_extension_file):
            missing.append(albedo_extension_file)
    return missing


def final_albedo_extension_file(native_final_file):
    """Return the final-extension albedo file matching a native-grid final CSV."""
    match = re.search(
        r'ssfr_simu_flux_(\d{8})_(\d+\.\d{3})-(\d+\.\d{3})_alt-(-?\d+\.\d+)km_final\.csv$',
        os.path.basename(native_final_file),
    )
    if match is None:
        return None
    date_s, time_start, time_end, alt = match.groups()
    return os.path.join(
        _fdir_general_,
        'sfc_alb',
        f'sfc_alb_{date_s}_{time_start}_{time_end}_{float(alt):.2f}km_final_extension.dat',
    )


def infer_final_iteration_from_native_final(native_final_file):
    """Infer the iteration number that was copied to one native-grid final CSV."""
    iteration_pattern = native_final_file.replace('_final.csv', '_iteration_*.csv')
    iteration_files = sorted(glob.glob(iteration_pattern))
    inferred_iters = []
    for iteration_file in iteration_files:
        match = re.search(r'_iteration_(\d+)\.csv$', iteration_file)
        if match is None:
            continue
        iter_value = int(match.group(1))
        try:
            if filecmp.cmp(native_final_file, iteration_file, shallow=False):
                return iter_value
        except OSError:
            pass
        inferred_iters.append(iter_value)
    if inferred_iters:
        return max(inferred_iters)
    return None


def max_limit_iteration(iterations, min_closure_iteration, max_additional_iterations):
    """Return the iteration that would trigger the configured max-iteration stop."""
    if not iterations:
        return None

    last_requested_iteration = max(iterations)
    if max_additional_iterations is None:
        return last_requested_iteration

    limit_iteration = min_closure_iteration + max_additional_iterations
    if limit_iteration in iterations:
        return limit_iteration
    return last_requested_iteration


def run_catalog_case(
    flt_trk_atm_corr,
    config,
    case_id,
    overwrite_lrt=True,
    iterations=range(3),
    closure_check=True,
    closure_thresholds=None,
    min_closure_iteration=2,
    max_additional_iterations=5,
    run_final_sim=True,
    skip_missing_cloud_observations=True,
):
    """Run one atmospheric-correction catalog case."""
    iterations = tuple(iterations)
    case = get_case(case_id)
    levels = catalog_case_levels(case)
    manual_cloud = case.get('manual_cloud', False)
    manual_cloud_cer = case.get('manual_cloud_cer', 0.0) or 0.0
    manual_cloud_cwp = case.get('manual_cloud_cwp', 0.0) or 0.0
    manual_cloud_cth = case.get('manual_cloud_cth', 0.0) or 0.0
    manual_cloud_cbh = case.get('manual_cloud_cbh', 0.0) or 0.0
    manual_cloud_cot = case.get('manual_cloud_cot', 0.0) or 0.0
    if levels is not None:
        level_text = ', '.join(f'{level:g}' for level in levels)
        print(
            f'{case_id}: using custom atmospheric levels '
            f'({len(levels)} levels, {levels[0]:.3f}-{levels[-1]:.3f} km): '
            f'[{level_text}]'
        )
    year, month, day = [int(part) for part in case['date'].split('-')]
    date_s = f'{year:04d}{month:02d}{day:02d}'
    missing_cloud_files = missing_cloud_observation_files(case, date_s)
    if missing_cloud_files:
        missing_text = '\n  '.join(missing_cloud_files)
        message = (
            f'{case_id}: missing {len(missing_cloud_files)} preprocessed cloud-observation file(s); '
            f'this case cannot run until they are generated. '
            f'Run python3 -m lrt_sim.ssfr_atm_corr.preprocess_runner {case_id} first.\n  {missing_text}'
        )
        if skip_missing_cloud_observations:
            print(f'{message}\n{case_id}: skipping case.')
            return False
        raise FileNotFoundError(message)

    def run_final_iteration(iter, final_status, final_overwrite_lrt=None):
        if not run_final_sim:
            return
        final_overwrite_lrt = overwrite_lrt if final_overwrite_lrt is None else final_overwrite_lrt
        flt_trk_atm_corr(
            date=datetime.datetime(year, month, day),
            tmhr_ranges_select=case['tmhr_ranges_select'],
            case_tag=case['case_tag'],
            config=config,
            levels=levels,
            simulation_interval=case['simulation_interval'],
            clear_sky=case['clear_sky'],
            overwrite_lrt=final_overwrite_lrt,
            manual_cloud=manual_cloud,
            manual_cloud_cer=manual_cloud_cer,
            manual_cloud_cwp=manual_cloud_cwp,
            manual_cloud_cth=manual_cloud_cth,
            manual_cloud_cbh=manual_cloud_cbh,
            manual_cloud_cot=manual_cloud_cot,
            iter=iter,
            final_sim=True,
            final_status=final_status,
        )

    native_final_files, missing_native_final_patterns = final_native_output_files(case, date_s)
    missing_final_extension_files = missing_final_extension_outputs(native_final_files)
    if run_final_sim and native_final_files and missing_final_extension_files and not missing_native_final_patterns:
        final_iter = infer_final_iteration_from_native_final(native_final_files[0])
        if final_iter is not None:
            print(
                f'{case_id}: found native final output(s) but missing '
                f'{len(missing_final_extension_files)} final-extension file(s); '
                f'running extension-only final simulation from iteration {final_iter}.'
            )
            run_final_iteration(
                final_iter,
                final_status='extension_only_from_existing_final',
                final_overwrite_lrt=False,
            )
            return True
        print(
            f'{case_id}: native final output(s) exist and final-extension file(s) are missing, '
            'but final iteration could not be inferred; continuing with normal iteration flow.'
        )

    last_iter = max_limit_iteration(iterations, min_closure_iteration, max_additional_iterations)
    if run_final_sim and last_iter is not None:
        last_iter_albedo_files, missing_last_iter_albedo_patterns = iteration_albedo_files(
            case,
            date_s,
            last_iter,
        )
        if last_iter_albedo_files and not missing_last_iter_albedo_patterns:
            print(
                f'{case_id}: found albedo file(s) for max-limit iteration {last_iter}; '
                'running final copy and final-extension simulation from that iteration.'
            )
            run_final_iteration(
                last_iter,
                final_status='max_iteration_from_existing_albedo',
            )
            return True
        if last_iter_albedo_files:
            print(
                f'{case_id}: found {len(last_iter_albedo_files)} albedo file(s) for '
                f'max-limit iteration {last_iter}, but missing '
                f'{len(missing_last_iter_albedo_patterns)} split range(s): '
                f'{summarize_missing_ranges(missing_last_iter_albedo_patterns)}. '
                'Continuing with normal iteration flow.'
            )

    for iter in iterations:
        flt_trk_atm_corr(
            date=datetime.datetime(year, month, day),
            tmhr_ranges_select=case['tmhr_ranges_select'],
            case_tag=case['case_tag'],
            config=config,
            levels=levels,
            simulation_interval=case['simulation_interval'],
            clear_sky=case['clear_sky'],
            overwrite_lrt=overwrite_lrt,
            manual_cloud=manual_cloud,
            manual_cloud_cer=manual_cloud_cer,
            manual_cloud_cwp=manual_cloud_cwp,
            manual_cloud_cth=manual_cloud_cth,
            manual_cloud_cbh=manual_cloud_cbh,
            manual_cloud_cot=manual_cloud_cot,
            iter=iter,
        )
        if closure_check and iter >= min_closure_iteration:
            output_files = iteration_output_files(case, date_s, iter)
            thresholds = closure_thresholds or DEFAULT_CLOSURE_THRESHOLDS
            print(f'{case_id}: closure check iteration {iter} found {len(output_files)} output file(s).')
            if iteration_closure_check(output_files, thresholds=closure_thresholds):
                print(f'{case_id}: iteration {iter} meets closure criteria; stopping further iterations.')
                run_final_iteration(iter, final_status='closure_passed')
                break
            if not output_files:
                print(f'{case_id}: no iteration {iter} output files found for closure check.')
            else:
                _, failed_mean_checks, mean_metrics = mean_closure_metric_status(output_files, thresholds)
                mean_metric_text = ', '.join(
                    f'mean_{name}={value:.5f}' for name, value in mean_metrics.items()
                )
                print(f'{case_id}: closure mean metrics: {mean_metric_text}')
                if failed_mean_checks:
                    print(f'{case_id}: closure mean FAIL: {"; ".join(failed_mean_checks)}')
                for output_file in output_files:
                    passed, failed_checks = closure_metric_status(output_file, thresholds)
                    if passed:
                        print(f'{case_id}: closure PASS {os.path.basename(output_file)}')
                    else:
                        print(
                            f'{case_id}: closure FAIL {os.path.basename(output_file)}: '
                            f'{"; ".join(failed_checks)}'
                        )

            additional_iterations = iter - min_closure_iteration
            max_iterations_reached = (
                max_additional_iterations is not None
                and additional_iterations >= max_additional_iterations
            )
            if max_iterations_reached:
                print(
                    f'{case_id}: iteration {iter} reached the max additional iteration limit '
                    f'({max_additional_iterations}) without closure; using this iteration for final output.'
                )
                run_final_iteration(iter, final_status='max_additional_iterations')
                break
    return True
