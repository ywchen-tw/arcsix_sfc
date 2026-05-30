"""Recovered date/time cases from the original SSFR atmospheric-correction script.

Most of these calls were commented out in the original script. This catalog keeps
the date, time-window, and case metadata available without bloating the driver.
Cases with ``has_custom_levels=True`` had a custom ``levels=...`` expression in
the original call; the full recovered call is preserved in ``original_call``.
"""

import datetime
import csv
import glob
import math
import os

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

ALL_CASE_CATALOG = [{'id': 'case_001',
  'date': '2024-05-31',
  'case_tag': 'clear_sky_track_1_atm_corr',
  'tmhr_ranges_select': [[15.689, 15.737],
                         [15.76, 15.776],
                         [15.855, 15.909],
                         [15.921, 16.076],
                         [16.088, 16.227],
                         [16.306, 16.313],
                         [16.319, 16.409],
                         [16.421, 16.475],
                         [16.501, 16.576],
                         [16.588, 16.715]],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': False,
  'manual_cloud_cer': 0.0,
  'manual_cloud_cwp': 0.0,
  'manual_cloud_cwp_expr': '0.0',
  'manual_cloud_cth': 0.0,
  'manual_cloud_cbh': 0.0,
  'manual_cloud_cot': 0.0,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 5, 31),\n'
                   '                        tmhr_ranges_select=[[15.689, 15.737], \n'
                   '                                            [15.760, 15.776],\n'
                   '                                            [15.855, 15.909],\n'
                   '                                            [15.921, 16.076],\n'
                   '                                            [16.088, 16.227],\n'
                   '                                            [16.306, 16.313],\n'
                   '                                            [16.319, 16.409],\n'
                   '                                            [16.421, 16.475],\n'
                   '                                            [16.501, 16.576],\n'
                   '                                            [16.588, 16.715]\n'
                   '                                            ],\n'
                   "                        case_tag='clear_sky_track_1_atm_corr',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=0.5,\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_atm=False,\n'
                   '                        overwrite_alb=False,\n'
                   '                        overwrite_lrt=True,\n'
                   '                        manual_cloud=False,\n'
                   '                        manual_cloud_cer=0.0,\n'
                   '                        manual_cloud_cwp=0.0,\n'
                   '                        manual_cloud_cth=0.0,\n'
                   '                        manual_cloud_cbh=0.0,\n'
                   '                        manual_cloud_cot=0.0,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_002',
  'date': '2024-08-07',
  'case_tag': 'clear_sky_track_atm_corr',
  'tmhr_ranges_select': [[17.39, 17.58]],
  'simulation_interval': 10,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': False,
  'manual_cloud_cer': None,
  'manual_cloud_cwp': None,
  'manual_cloud_cwp_expr': None,
  'manual_cloud_cth': None,
  'manual_cloud_cbh': None,
  'manual_cloud_cot': None,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 8, 7),\n'
                   '                        tmhr_ranges_select=[\n'
                   '                                            [17.39, 17.58],\n'
                   '                                            ],\n'
                   "                        case_tag='clear_sky_track_atm_corr',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=10,\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_lrt=True,\n'
                   '                        manual_cloud=False,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_003',
  'date': '2024-05-28',
  'case_tag': 'clear_sky_track_atm_corr',
  'tmhr_ranges_select': [[12.62, 15.18]],
  'simulation_interval': 15,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': False,
  'manual_cloud_cer': None,
  'manual_cloud_cwp': None,
  'manual_cloud_cwp_expr': None,
  'manual_cloud_cth': None,
  'manual_cloud_cbh': None,
  'manual_cloud_cot': None,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 5, 28),\n'
                   '                        tmhr_ranges_select=[\n'
                   '                                            [12.62, 15.18],\n'
                   '                                            ],\n'
                   "                        case_tag='clear_sky_track_atm_corr',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=15,\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_lrt=True,\n'
                   '                        manual_cloud=False,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_004',
  'date': '2024-05-30',
  'case_tag': 'clear_sky_track_atm_corr',
  'tmhr_ranges_select': [[11.3, 12.29], [12.4, 12.79], [16.38, 17.42]],
  'simulation_interval': 15,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': False,
  'manual_cloud_cer': None,
  'manual_cloud_cwp': None,
  'manual_cloud_cwp_expr': None,
  'manual_cloud_cth': None,
  'manual_cloud_cbh': None,
  'manual_cloud_cot': None,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 5, 30),\n'
                   '                        tmhr_ranges_select=[\n'
                   '                                            [11.30, 12.29],\n'
                   '                                            [12.40, 12.79],\n'
                   '                                            [16.38, 17.42],\n'
                   '                                            ],\n'
                   "                        case_tag='clear_sky_track_atm_corr',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=15,\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_lrt=True,\n'
                   '                        manual_cloud=False,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_005',
  'date': '2024-05-31',
  'case_tag': 'clear_sky_track_atm_corr',
  'tmhr_ranges_select': [[12.77, 13.04], [13.2, 13.55], [14.5, 15.04], [16.89, 17.43]],
  'simulation_interval': 15,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': False,
  'manual_cloud_cer': None,
  'manual_cloud_cwp': None,
  'manual_cloud_cwp_expr': None,
  'manual_cloud_cth': None,
  'manual_cloud_cbh': None,
  'manual_cloud_cot': None,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 5, 31),\n'
                   '                        tmhr_ranges_select=[\n'
                   '                                            [12.77, 13.04],\n'
                   '                                            [13.20, 13.55],\n'
                   '                                            [14.50, 15.04],\n'
                   '                                            [16.89, 17.43],\n'
                   '                                            \n'
                   '                                            ],\n'
                   "                        case_tag='clear_sky_track_atm_corr',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=15,\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_lrt=True,\n'
                   '                        manual_cloud=False,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_006',
  'date': '2024-06-03',
  'case_tag': 'clear_sky_track_atm_corr',
  'tmhr_ranges_select': [[11.29, 11.86], [11.87, 13.23], [13.23, 13.44], [16.38, 17.8]],
  'simulation_interval': 15,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': False,
  'manual_cloud_cer': None,
  'manual_cloud_cwp': None,
  'manual_cloud_cwp_expr': None,
  'manual_cloud_cth': None,
  'manual_cloud_cbh': None,
  'manual_cloud_cot': None,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 6, 3),\n'
                   '                        tmhr_ranges_select=[\n'
                   '                                            [11.29, 11.86],\n'
                   '                                            [11.87, 13.23],\n'
                   '                                            [13.23, 13.44],\n'
                   '                                            [16.38, 17.80],\n'
                   '                                            \n'
                   '                                            ],\n'
                   "                        case_tag='clear_sky_track_atm_corr',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=15,\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_lrt=True,\n'
                   '                        manual_cloud=False,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_007',
  'date': '2024-06-05',
  'case_tag': 'clear_sky_track_atm_corr',
  'tmhr_ranges_select': [[12.0, 12.2]],
  'simulation_interval': 15,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': False,
  'manual_cloud_cer': None,
  'manual_cloud_cwp': None,
  'manual_cloud_cwp_expr': None,
  'manual_cloud_cth': None,
  'manual_cloud_cbh': None,
  'manual_cloud_cot': None,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 6, 5),\n'
                   '                        tmhr_ranges_select=[\n'
                   '                                            # [11.33, 11.88],\n'
                   '                                            [12.00, 12.20],\n'
                   '                                            # [12.33, 13.80],\n'
                   '                                            ],\n'
                   "                        case_tag='clear_sky_track_atm_corr',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=15,\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_lrt=True,\n'
                   '                        manual_cloud=False,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_008',
  'date': '2024-06-06',
  'case_tag': 'clear_sky_track_atm_corr',
  'tmhr_ranges_select': [[11.29, 11.4]],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': False,
  'manual_cloud_cer': None,
  'manual_cloud_cwp': None,
  'manual_cloud_cwp_expr': None,
  'manual_cloud_cth': None,
  'manual_cloud_cbh': None,
  'manual_cloud_cot': None,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 6, 6),\n'
                   '                        tmhr_ranges_select=[\n'
                   '                                            # [11.29, 13.31],\n'
                   '                                            # [17.26, 18.32],\n'
                   '                                            [11.29, 11.40],\n'
                   '                                            ],\n'
                   "                        case_tag='clear_sky_track_atm_corr',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=0.5,\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_lrt=True,\n'
                   '                        manual_cloud=False,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_009',
  'date': '2024-06-07',
  'case_tag': 'clear_sky_track_atm_corr',
  'tmhr_ranges_select': [[13.61, 14.1], [14.17, 14.3], [14.6, 14.92], [17.67, 18.25], [18.33, 18.52]],
  'simulation_interval': 15,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': False,
  'manual_cloud_cer': None,
  'manual_cloud_cwp': None,
  'manual_cloud_cwp_expr': None,
  'manual_cloud_cth': None,
  'manual_cloud_cbh': None,
  'manual_cloud_cot': None,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 6, 7),\n'
                   '                        tmhr_ranges_select=[\n'
                   '                                            [13.61, 14.10],\n'
                   '                                            [14.17, 14.30],\n'
                   '                                            [14.60, 14.92],\n'
                   '                                            [17.67, 18.25],\n'
                   '                                            [18.33, 18.52]\n'
                   '                                            ],\n'
                   "                        case_tag='clear_sky_track_atm_corr',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=15,\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_lrt=True,\n'
                   '                        manual_cloud=False,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_010',
  'date': '2024-06-10',
  'case_tag': 'clear_sky_track_atm_corr',
  'tmhr_ranges_select': [[11.28, 11.51]],
  'simulation_interval': 15,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': False,
  'manual_cloud_cer': None,
  'manual_cloud_cwp': None,
  'manual_cloud_cwp_expr': None,
  'manual_cloud_cth': None,
  'manual_cloud_cbh': None,
  'manual_cloud_cot': None,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 6, 10),\n'
                   '                        tmhr_ranges_select=[\n'
                   '                                            [11.28, 11.51],\n'
                   '                                            ],\n'
                   "                        case_tag='clear_sky_track_atm_corr',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=15,\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_lrt=True,\n'
                   '                        manual_cloud=False,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_011',
  'date': '2024-06-11',
  'case_tag': 'clear_sky_track_atm_corr',
  'tmhr_ranges_select': [[11.28, 11.51]],
  'simulation_interval': 15,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': False,
  'manual_cloud_cer': None,
  'manual_cloud_cwp': None,
  'manual_cloud_cwp_expr': None,
  'manual_cloud_cth': None,
  'manual_cloud_cbh': None,
  'manual_cloud_cot': None,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 6, 11),\n'
                   '                        tmhr_ranges_select=[\n'
                   '                                            [11.28, 11.51],\n'
                   '                                            ],\n'
                   "                        case_tag='clear_sky_track_atm_corr',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=15,\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_lrt=True,\n'
                   '                        manual_cloud=False,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_012',
  'date': '2024-08-09',
  'case_tag': 'clear_sky_track_atm_corr',
  'tmhr_ranges_select': [[17.7, 17.87]],
  'simulation_interval': 6,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': False,
  'manual_cloud_cer': None,
  'manual_cloud_cwp': None,
  'manual_cloud_cwp_expr': None,
  'manual_cloud_cth': None,
  'manual_cloud_cbh': None,
  'manual_cloud_cot': None,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 8, 9),\n'
                   '                        tmhr_ranges_select=[\n'
                   '                                            [17.70, 17.87],\n'
                   '                                            ],\n'
                   "                        case_tag='clear_sky_track_atm_corr',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=6,\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_lrt=True,\n'
                   '                        manual_cloud=False,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_013',
  'date': '2024-07-29',
  'case_tag': 'clear_sky_track_atm_corr',
  'tmhr_ranges_select': [[13.05, 13.45]],
  'simulation_interval': 10,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': False,
  'manual_cloud_cer': None,
  'manual_cloud_cwp': None,
  'manual_cloud_cwp_expr': None,
  'manual_cloud_cth': None,
  'manual_cloud_cbh': None,
  'manual_cloud_cot': None,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 7, 29),\n'
                   '                        tmhr_ranges_select=[\n'
                   '                                            [13.05, 13.45],\n'
                   '                                            ],\n'
                   "                        case_tag='clear_sky_track_atm_corr',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=10,\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_lrt=True,\n'
                   '                        manual_cloud=False,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_014',
  'date': '2024-06-06',
  'case_tag': 'cloudy_track_4_atm_corr_before',
  'tmhr_ranges_select': [[13.99, 14.18], [14.26, 14.46]],
  'simulation_interval': 10,
  'clear_sky': False,
  'manual_cloud': True,
  'has_custom_levels': True,
  'manual_cloud_cer': 6.9,
  'manual_cloud_cwp': 0.0231,
  'manual_cloud_cwp_expr': '0.0231',
  'manual_cloud_cth': 0.3,
  'manual_cloud_cbh': 0.101,
  'manual_cloud_cot': 5.01,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 6, 6),\n'
                   '                        tmhr_ranges_select=[[13.99, 14.18], [14.26, 14.46]],\n'
                   "                        case_tag='cloudy_track_4_atm_corr_before',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=10,\n'
                   '                        levels=np.concatenate((np.arange(0.0, 1.61, 0.1),\n'
                   '                                            np.array([1.8, 2.0, 2.5, 3.0, 4.0]), \n'
                   '                                            np.arange(5.0, 10.1, 2.5),\n'
                   '                                            np.array([15, 20, 30., 40., 45.]))),\n'
                   '                        clear_sky=False,\n'
                   '                        overwrite_atm=False,\n'
                   '                        overwrite_alb=False,\n'
                   '                        overwrite_lrt=True,\n'
                   '                        manual_cloud=True,\n'
                   '                        manual_cloud_cer=6.9,\n'
                   '                        manual_cloud_cwp=0.0231,\n'
                   '                        manual_cloud_cth=0.3,\n'
                   '                        manual_cloud_cbh=0.101,\n'
                   '                        manual_cloud_cot=5.01,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_015',
  'date': '2024-06-07',
  'case_tag': 'cloudy_track_2_atm_corr',
  'tmhr_ranges_select': [[15.34, 15.7583], [15.8403, 16.2653]],
  'simulation_interval': 0.5,
  'clear_sky': False,
  'manual_cloud': True,
  'has_custom_levels': True,
  'manual_cloud_cer': 8.0,
  'manual_cloud_cwp': 0.0229,
  'manual_cloud_cwp_expr': '0.0229',
  'manual_cloud_cth': 0.47,
  'manual_cloud_cbh': 0.25,
  'manual_cloud_cot': 4.3,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 6, 7),\n'
                   '                        tmhr_ranges_select=[[15.3400, 15.7583], [15.8403, 16.2653]],\n'
                   "                        case_tag='cloudy_track_2_atm_corr',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=0.5,\n'
                   '                        levels=np.concatenate((np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, '
                   '0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 1.0]),\n'
                   '                                            np.array([1.5, 2.0, 3.0, 4.0]), \n'
                   '                                            np.arange(5.0, 10.1, 2.5),\n'
                   '                                            np.array([15, 20, 30., 40., 45.]))),\n'
                   '                        clear_sky=False,\n'
                   '                        overwrite_atm=False,\n'
                   '                        overwrite_alb=False,\n'
                   '                        overwrite_lrt=True,\n'
                   '                        manual_cloud=True,\n'
                   '                        manual_cloud_cer=8.0,\n'
                   '                        manual_cloud_cwp=0.0229,\n'
                   '                        manual_cloud_cth=0.47,\n'
                   '                        manual_cloud_cbh=0.25,\n'
                   '                        manual_cloud_cot=4.3,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_016',
  'date': '2024-06-11',
  'case_tag': 'cloudy_track_1_atm_corr',
  'tmhr_ranges_select': [[16.076, 16.109], [16.123, 16.255]],
  'simulation_interval': 0.5,
  'clear_sky': False,
  'manual_cloud': True,
  'has_custom_levels': True,
  'manual_cloud_cer': 3.4,
  'manual_cloud_cwp': 0.03209,
  'manual_cloud_cwp_expr': '0.03209',
  'manual_cloud_cth': 1.678,
  'manual_cloud_cbh': 1.262,
  'manual_cloud_cot': 14.173,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 6, 11),\n'
                   '                        tmhr_ranges_select=[[16.076, 16.109],\n'
                   '                                            [16.123, 16.255]],\n'
                   "                        case_tag='cloudy_track_1_atm_corr',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=0.5,\n'
                   '                        levels=np.concatenate((np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, '
                   '0.9, 1.0]),\n'
                   '                                            np.array([1.2, 1.4, 1.7, 2.0, 2.5, 3.0, 4.0]), \n'
                   '                                            np.arange(5.0, 10.1, 2.5),\n'
                   '                                            np.array([15, 20, 30., 40., 45.]))),\n'
                   '                        clear_sky=False,\n'
                   '                        overwrite_atm=True,\n'
                   '                        overwrite_alb=False,\n'
                   '                        overwrite_lrt=True,\n'
                   '                        manual_cloud=True,\n'
                   '                        manual_cloud_cer=3.4,\n'
                   '                        manual_cloud_cwp=0.03209,\n'
                   '                        manual_cloud_cth=1.678,\n'
                   '                        manual_cloud_cbh=1.262,\n'
                   '                        manual_cloud_cot=14.173,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_017',
  'date': '2024-06-13',
  'case_tag': 'cloudy_track_1_atm_corr',
  'tmhr_ranges_select': [[15.85, 15.882], [16.057, 16.06]],
  'simulation_interval': 0.5,
  'clear_sky': False,
  'manual_cloud': True,
  'has_custom_levels': True,
  'manual_cloud_cer': 13.4,
  'manual_cloud_cwp': 0.08572,
  'manual_cloud_cwp_expr': '0.08572',
  'manual_cloud_cth': 0.637,
  'manual_cloud_cbh': 0.119,
  'manual_cloud_cot': 9.57,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 6, 13),\n'
                   '                        tmhr_ranges_select=[[15.85, 15.882], [16.057, 16.060]],\n'
                   "                        case_tag='cloudy_track_1_atm_corr',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=0.5,\n'
                   '                        levels=np.concatenate((np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, '
                   '0.9, 1.0]),\n'
                   '                                            np.array([1.2, 1.4, 1.7, 2.0, 2.5, 3.0, 4.0]), \n'
                   '                                            np.arange(5.0, 10.1, 2.5),\n'
                   '                                            np.array([15, 20, 30., 40., 45.]))),\n'
                   '                        clear_sky=False,\n'
                   '                        overwrite_atm=False,\n'
                   '                        overwrite_alb=False,\n'
                   '                        overwrite_lrt=True,\n'
                   '                        manual_cloud=True,\n'
                   '                        manual_cloud_cer=13.4,\n'
                   '                        manual_cloud_cwp=0.08572,\n'
                   '                        manual_cloud_cth=0.637,\n'
                   '                        manual_cloud_cbh=0.119,\n'
                   '                        manual_cloud_cot=9.57,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_018',
  'date': '2024-06-13',
  'case_tag': 'cloudy_track_2_atm_corr',
  'tmhr_ranges_select': [[15.85, 15.882], [16.057, 16.06]],
  'simulation_interval': 0.5,
  'clear_sky': False,
  'manual_cloud': True,
  'has_custom_levels': True,
  'manual_cloud_cer': 22.5,
  'manual_cloud_cwp': 0.03711,
  'manual_cloud_cwp_expr': '0.03711',
  'manual_cloud_cth': 0.919,
  'manual_cloud_cbh': 0.609,
  'manual_cloud_cot': 2.48,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 6, 13),\n'
                   '                        tmhr_ranges_select=[[15.85, 15.882], [16.057, 16.060]],\n'
                   "                        case_tag='cloudy_track_2_atm_corr',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=0.5,\n'
                   '                        levels=np.concatenate((np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, '
                   '0.9, 1.0]),\n'
                   '                                            np.array([1.2, 1.4, 1.7, 2.0, 2.5, 3.0, 4.0]), \n'
                   '                                            np.arange(5.0, 10.1, 2.5),\n'
                   '                                            np.array([15, 20, 30., 40., 45.]))),\n'
                   '                        clear_sky=False,\n'
                   '                        overwrite_atm=False,\n'
                   '                        overwrite_alb=False,\n'
                   '                        overwrite_lrt=True,\n'
                   '                        manual_cloud=True,\n'
                   '                        manual_cloud_cer=22.5,\n'
                   '                        manual_cloud_cwp=0.03711,\n'
                   '                        manual_cloud_cth=0.919,\n'
                   '                        manual_cloud_cbh=0.609,\n'
                   '                        manual_cloud_cot=2.48,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_019',
  'date': '2024-06-13',
  'case_tag': 'cloudy_track_3_atm_corr',
  'tmhr_ranges_select': [[16.0555, 16.0585], [16.207, 16.213]],
  'simulation_interval': 0.5,
  'clear_sky': False,
  'manual_cloud': True,
  'has_custom_levels': True,
  'manual_cloud_cer': 12.5,
  'manual_cloud_cwp': 0.03308,
  'manual_cloud_cwp_expr': '0.03308',
  'manual_cloud_cth': 1.023,
  'manual_cloud_cbh': 0.677,
  'manual_cloud_cot': 3.98,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 6, 13),\n'
                   '                        tmhr_ranges_select=[[16.0555, 16.0585], [16.207, 16.213]],\n'
                   "                        case_tag='cloudy_track_3_atm_corr',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=0.5,\n'
                   '                        levels=np.concatenate((np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, '
                   '0.9, 1.0, 1.1]),\n'
                   '                                            np.array([1.2, 1.4, 1.7, 2.0, 2.5, 3.0, 4.0]), \n'
                   '                                            np.arange(5.0, 10.1, 2.5),\n'
                   '                                            np.array([15, 20, 30., 40., 45.]))),\n'
                   '                        clear_sky=False,\n'
                   '                        overwrite_atm=False,\n'
                   '                        overwrite_alb=False,\n'
                   '                        overwrite_lrt=True,\n'
                   '                        manual_cloud=True,\n'
                   '                        manual_cloud_cer=12.5,\n'
                   '                        manual_cloud_cwp=0.03308,\n'
                   '                        manual_cloud_cth=1.023,\n'
                   '                        manual_cloud_cbh=0.677,\n'
                   '                        manual_cloud_cot=3.98,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_020',
  'date': '2024-06-13',
  'case_tag': 'clear_track_1_atm_corr',
  'tmhr_ranges_select': [[16.557, 16.58],
                         [16.591, 16.64],
                         [16.656, 16.74],
                         [16.907, 16.962],
                         [16.972, 16.976],
                         [16.989, 16.995],
                         [17.017, 17.026],
                         [17.067, 17.142],
                         [17.156, 17.206],
                         [17.375, 17.405]],
  'simulation_interval': 15,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': True,
  'manual_cloud_cer': 0.0,
  'manual_cloud_cwp': 0.0,
  'manual_cloud_cwp_expr': '0.0',
  'manual_cloud_cth': 0.0,
  'manual_cloud_cbh': 0.0,
  'manual_cloud_cot': 0.0,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 6, 13),\n'
                   '                        tmhr_ranges_select=[\n'
                   '                                            [16.557, 16.580], \n'
                   '                                            [16.591, 16.640], \n'
                   '                                            [16.656, 16.740],\n'
                   '                                            [16.907, 16.962],\n'
                   '                                            [16.972, 16.976],\n'
                   '                                            [16.989, 16.995],\n'
                   '                                            [17.017, 17.026],\n'
                   '                                            [17.067, 17.142],\n'
                   '                                            [17.156, 17.206],\n'
                   '                                            [17.375, 17.405],\n'
                   '                                            ],\n'
                   "                        case_tag='clear_track_1_atm_corr',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=15,\n'
                   '                        levels=np.concatenate((np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, '
                   '0.9, 1.0,]),\n'
                   '                                            np.array([1.2, 1.4, 1.7, 2.0, 2.5, 3.0, 4.0]), \n'
                   '                                            np.arange(5.0, 10.1, 2.5),\n'
                   '                                            np.array([15, 20, 30., 40., 45.]))),\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_lrt=True,\n'
                   '                        manual_cloud=False,\n'
                   '                        manual_cloud_cer=0.0,\n'
                   '                        manual_cloud_cwp=0.0,\n'
                   '                        manual_cloud_cth=0.0,\n'
                   '                        manual_cloud_cbh=0.0,\n'
                   '                        manual_cloud_cot=0.0,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_021',
  'date': '2024-06-05',
  'case_tag': 'clear_sky_track_atm_corr',
  'tmhr_ranges_select': [[14.594, 14.747],
                         [14.76, 14.913],
                         [14.926, 15.062],
                         [15.56, 15.58],
                         [15.593, 15.746],
                         [15.76, 15.912],
                         [16.05, 16.08],
                         [16.093, 16.247],
                         [16.26, 16.413]],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': False,
  'manual_cloud_cer': 0.0,
  'manual_cloud_cwp': 0.0,
  'manual_cloud_cwp_expr': '0.0',
  'manual_cloud_cth': 0.0,
  'manual_cloud_cbh': 0.0,
  'manual_cloud_cot': 0.0,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 6, 5),\n'
                   '                        tmhr_ranges_select=[\n'
                   '                                            [14.594, 14.747],\n'
                   '                                            [14.760, 14.913], # cloud probably\n'
                   '                                            [14.926, 15.062], # cloud probably\n'
                   '                                            [15.560, 15.580],\n'
                   '                                            [15.593, 15.746],\n'
                   '                                            [15.760, 15.912],\n'
                   '                                            [16.050, 16.080],\n'
                   '                                            [16.093, 16.247], # cloud shadow\n'
                   '                                            [16.260, 16.413], # cloud shadow\n'
                   '                                            ],\n'
                   "                        case_tag='clear_sky_track_atm_corr',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=0.5,\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_atm=False,\n'
                   '                        overwrite_alb=False,\n'
                   '                        overwrite_lrt=True,\n'
                   '                        manual_cloud=False,\n'
                   '                        manual_cloud_cer=0.0,\n'
                   '                        manual_cloud_cwp=0.0,\n'
                   '                        manual_cloud_cth=0.0,\n'
                   '                        manual_cloud_cbh=0.0,\n'
                   '                        manual_cloud_cot=0.0,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_022',
  'date': '2024-06-06',
  'case_tag': 'clear_sky_track_atm_corr_before',
  'tmhr_ranges_select': [[16.251, 16.28], [16.293, 16.325], [16.704, 16.78]],
  'simulation_interval': 20,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': False,
  'manual_cloud_cer': 0.0,
  'manual_cloud_cwp': 0.0,
  'manual_cloud_cwp_expr': '0.0',
  'manual_cloud_cth': 0.0,
  'manual_cloud_cbh': 0.0,
  'manual_cloud_cot': 0.0,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 6, 6),\n'
                   '                        tmhr_ranges_select=[\n'
                   '                                            [16.251, 16.280], \n'
                   '                                            [16.293, 16.325],\n'
                   '                                            [16.704, 16.780],\n'
                   '                                            ],\n'
                   "                        case_tag='clear_sky_track_atm_corr_before',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=20, # in minute\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_lrt=True,\n'
                   '                        manual_cloud=False,\n'
                   '                        manual_cloud_cer=0.0,\n'
                   '                        manual_cloud_cwp=0.0,\n'
                   '                        manual_cloud_cth=0.0,\n'
                   '                        manual_cloud_cbh=0.0,\n'
                   '                        manual_cloud_cot=0.0,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_023',
  'date': '2024-06-06',
  'case_tag': 'clear_sky_track_2_atm_corr_after',
  'tmhr_ranges_select': [[12.84, 12.92], [16.86, 16.93], [17.03, 17.09], [17.31, 17.41], [17.63, 17.69]],
  'simulation_interval': 10,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': False,
  'manual_cloud_cer': 0.0,
  'manual_cloud_cwp': 0.0,
  'manual_cloud_cwp_expr': '0.0',
  'manual_cloud_cth': 0.0,
  'manual_cloud_cbh': 0.0,
  'manual_cloud_cot': 0.0,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 6, 6),\n'
                   '                        tmhr_ranges_select=[\n'
                   '                                            [12.84, 12.92],\n'
                   '                                            [16.86, 16.93],\n'
                   '                                            [17.03, 17.09],\n'
                   '                                            [17.31, 17.41],\n'
                   '                                            [17.63, 17.69],\n'
                   '                                            ],\n'
                   "                        case_tag='clear_sky_track_2_atm_corr_after',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=10, # in minute\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_atm=False,\n'
                   '                        overwrite_alb=False,\n'
                   '                        overwrite_lrt=True,\n'
                   '                        manual_cloud=False,\n'
                   '                        manual_cloud_cer=0.0,\n'
                   '                        manual_cloud_cwp=0.0,\n'
                   '                        manual_cloud_cth=0.0,\n'
                   '                        manual_cloud_cbh=0.0,\n'
                   '                        manual_cloud_cot=0.0,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_024',
  'date': '2024-06-13',
  'case_tag': 'clear_sky_spiral_atm_corr',
  'tmhr_ranges_select': [[13.0194, 13.0569],
                         [13.0792, 13.0937],
                         [13.1153, 13.1306],
                         [13.1569, 13.1653],
                         [13.1944, 13.2069],
                         [13.2319, 13.2514],
                         [13.2736, 13.2889],
                         [13.3125, 13.3278],
                         [13.35, 13.3708],
                         [13.3889, 13.4208],
                         [13.4417, 13.4708],
                         [13.5181, 13.5667]],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': False,
  'manual_cloud_cer': 0.0,
  'manual_cloud_cwp': 0.0,
  'manual_cloud_cwp_expr': '0.0',
  'manual_cloud_cth': 0.0,
  'manual_cloud_cbh': 0.0,
  'manual_cloud_cot': 0.0,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 6, 13), # more popcorn clouds\n'
                   '                        tmhr_ranges_select=[[13.0194, 13.0569],\n'
                   '                                            [13.0792, 13.0937],\n'
                   '                                            [13.1153, 13.1306],\n'
                   '                                            [13.1569, 13.1653],\n'
                   '                                            [13.1944, 13.2069],\n'
                   '                                            [13.2319, 13.2514],\n'
                   '                                            [13.2736, 13.2889],\n'
                   '                                            [13.3125, 13.3278],\n'
                   '                                            [13.3500, 13.3708],\n'
                   '                                            [13.3889, 13.4208],\n'
                   '                                            [13.4417, 13.4708],\n'
                   '                                            [13.5181, 13.5667], # below clouds?\n'
                   '                                            ],\n'
                   "                        case_tag='clear_sky_spiral_atm_corr',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=0.5,\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_atm=False,\n'
                   '                        overwrite_alb=False,\n'
                   '                        overwrite_lrt=True,\n'
                   '                        manual_cloud=False,\n'
                   '                        manual_cloud_cer=0.0,\n'
                   '                        manual_cloud_cwp=0.0,\n'
                   '                        manual_cloud_cth=0.0,\n'
                   '                        manual_cloud_cbh=0.0,\n'
                   '                        manual_cloud_cot=0.0,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_025',
  'date': '2024-06-06',
  'case_tag': 'clear_sky_spiral_atm_corr',
  'tmhr_ranges_select': [[17.0833, 17.0986],
                         [17.1264, 17.1333],
                         [17.1542, 17.1601],
                         [17.1833, 17.1931],
                         [17.2153, 17.2181],
                         [17.2403, 17.25]],
  'simulation_interval': 10,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': False,
  'manual_cloud_cer': 0.0,
  'manual_cloud_cwp': 0.0,
  'manual_cloud_cwp_expr': '0.0',
  'manual_cloud_cth': 0.0,
  'manual_cloud_cbh': 0.0,
  'manual_cloud_cot': 0.0,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 6, 6),\n'
                   '                        tmhr_ranges_select=[[17.0833, 17.0986],\n'
                   '                                            [17.1264, 17.1333],\n'
                   '                                            [17.1542, 17.1601],\n'
                   '                                            [17.1833, 17.1931],\n'
                   '                                            [17.2153, 17.2181],\n'
                   '                                            [17.2403, 17.2500],\n'
                   '                                            ],\n'
                   "                        case_tag='clear_sky_spiral_atm_corr',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=10,\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_atm=False,\n'
                   '                        overwrite_alb=False,\n'
                   '                        overwrite_lrt=True,\n'
                   '                        manual_cloud=False,\n'
                   '                        manual_cloud_cer=0.0,\n'
                   '                        manual_cloud_cwp=0.0,\n'
                   '                        manual_cloud_cth=0.0,\n'
                   '                        manual_cloud_cbh=0.0,\n'
                   '                        manual_cloud_cot=0.0,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_026',
  'date': '2024-06-05',
  'case_tag': 'clear_sky_spiral_atm_corr',
  'tmhr_ranges_select': [[13.7889, 13.8097],
                         [13.8347, 13.85],
                         [13.8764, 13.8903],
                         [13.9236, 13.9264],
                         [13.9389, 13.9403],
                         [13.9528, 13.9722],
                         [13.9958, 14.0153],
                         [14.0417, 14.0597],
                         [14.0819, 14.1],
                         [14.1264, 14.1542],
                         [14.1762, 14.2],
                         [14.2194, 14.2444],
                         [14.2597, 14.2833]],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': False,
  'manual_cloud_cer': 0.0,
  'manual_cloud_cwp': 0.0,
  'manual_cloud_cwp_expr': '0.0',
  'manual_cloud_cth': 0.0,
  'manual_cloud_cbh': 0.0,
  'manual_cloud_cot': 0.0,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 6, 5),\n'
                   '                        tmhr_ranges_select=[[13.7889, 13.8097],\n'
                   '                                            [13.8347, 13.8500],\n'
                   '                                            [13.8764, 13.8903],\n'
                   '                                            [13.9236, 13.9264],\n'
                   '                                            [13.9389, 13.9403],\n'
                   '                                            [13.9528, 13.9722],\n'
                   '                                            [13.9958, 14.0153],\n'
                   '                                            [14.0417, 14.0597],\n'
                   '                                            [14.0819, 14.1000],\n'
                   '                                            [14.1264, 14.1542],\n'
                   '                                            [14.1762, 14.2000],\n'
                   '                                            [14.2194, 14.2444],\n'
                   '                                            [14.2597, 14.2833]\n'
                   '                                            ],\n'
                   "                        case_tag='clear_sky_spiral_atm_corr',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=0.5,\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_atm=False,\n'
                   '                        overwrite_alb=False,\n'
                   '                        overwrite_lrt=True,\n'
                   '                        manual_cloud=False,\n'
                   '                        manual_cloud_cer=0.0,\n'
                   '                        manual_cloud_cwp=0.0,\n'
                   '                        manual_cloud_cth=0.0,\n'
                   '                        manual_cloud_cbh=0.0,\n'
                   '                        manual_cloud_cot=0.0,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_027',
  'date': '2024-06-05',
  'case_tag': 'clear_sky_spiral_atm_corr_after_corr_R3_v2',
  'tmhr_ranges_select': [[13.7889, 13.801],
                         [13.835, 13.8395],
                         [13.878, 13.8885],
                         [13.924, 13.9255],
                         [13.954, 13.9715],
                         [13.998, 14.0153],
                         [14.0417, 14.0475],
                         [14.056, 14.059],
                         [14.0825, 14.0975],
                         [14.1264, 14.1525],
                         [14.1762, 14.1975],
                         [14.2194, 14.242],
                         [14.2605, 14.281]],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': False,
  'manual_cloud_cer': 0.0,
  'manual_cloud_cwp': 0.0,
  'manual_cloud_cwp_expr': '0.0',
  'manual_cloud_cth': 0.0,
  'manual_cloud_cbh': 0.0,
  'manual_cloud_cot': 0.0,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 6, 5),\n'
                   '                        tmhr_ranges_select=[[13.7889, 13.8010],\n'
                   '                                            [13.8350, 13.8395],\n'
                   '                                            [13.8780, 13.8885],\n'
                   '                                            [13.9240, 13.9255],\n'
                   '                                            # [13.9389, 13.9403],\n'
                   '                                            [13.9540, 13.9715],\n'
                   '                                            [13.9980, 14.0153],\n'
                   '                                            # [14.0417, 14.0575],\n'
                   '                                            [14.0417, 14.0475],\n'
                   '                                            [14.0560, 14.0590],\n'
                   '                                            [14.0825, 14.0975],\n'
                   '                                            [14.1264, 14.1525],\n'
                   '                                            [14.1762, 14.1975],\n'
                   '                                            [14.2194, 14.2420],\n'
                   '                                            [14.2605, 14.2810]\n'
                   '                                            ],\n'
                   "                        case_tag='clear_sky_spiral_atm_corr_after_corr_R3_v2',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=0.5,\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_atm=False,\n'
                   '                        overwrite_alb=False,\n'
                   '                        overwrite_lrt=True,\n'
                   '                        manual_cloud=False,\n'
                   '                        manual_cloud_cer=0.0,\n'
                   '                        manual_cloud_cwp=0.0,\n'
                   '                        manual_cloud_cth=0.0,\n'
                   '                        manual_cloud_cbh=0.0,\n'
                   '                        manual_cloud_cot=0.0,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_028',
  'date': '2024-06-05',
  'case_tag': 'clear_sky_spiral_atm_corr_R1',
  'tmhr_ranges_select': [[13.7889, 13.801],
                         [13.835, 13.8395],
                         [13.878, 13.8885],
                         [13.954, 13.9715],
                         [13.998, 14.0153],
                         [14.0417, 14.0475],
                         [14.056, 14.059],
                         [14.0825, 14.0975],
                         [14.1264, 14.1525],
                         [14.1762, 14.1975],
                         [14.2194, 14.242],
                         [14.2605, 14.281]],
  'simulation_interval': None,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': False,
  'manual_cloud_cer': 0.0,
  'manual_cloud_cwp': 0.0,
  'manual_cloud_cwp_expr': '0.0',
  'manual_cloud_cth': 0.0,
  'manual_cloud_cbh': 0.0,
  'manual_cloud_cot': 0.0,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 6, 5),\n'
                   '                        tmhr_ranges_select=[\n'
                   '                                            [13.7889, 13.8010],\n'
                   '                                            [13.8350, 13.8395],\n'
                   '                                            [13.8780, 13.8885],\n'
                   '                                            # [13.9240, 13.9255],\n'
                   '                                            # [13.9389, 13.9403],\n'
                   '                                            [13.9540, 13.9715],\n'
                   '                                            [13.9980, 14.0153],\n'
                   '                                            # [14.0417, 14.0575],\n'
                   '                                            [14.0417, 14.0475],\n'
                   '                                            [14.0560, 14.0590],\n'
                   '                                            [14.0825, 14.0975],\n'
                   '                                            [14.1264, 14.1525],\n'
                   '                                            [14.1762, 14.1975],\n'
                   '                                            [14.2194, 14.2420],\n'
                   '                                            [14.2605, 14.2810]\n'
                   '                                            ],\n'
                   "                        case_tag='clear_sky_spiral_atm_corr_R1',\n"
                   '                        config=config,\n'
                   '                        # simulation_interval=50,\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_lrt=True,\n'
                   '                        manual_cloud=False,\n'
                   '                        manual_cloud_cer=0.0,\n'
                   '                        manual_cloud_cwp=0.0,\n'
                   '                        manual_cloud_cth=0.0,\n'
                   '                        manual_cloud_cbh=0.0,\n'
                   '                        manual_cloud_cot=0.0,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_029',
  'date': '2024-05-28',
  'case_tag': 'clear_atm_corr',
  'tmhr_ranges_select': [[15.61, 15.822], [16.905, 17.404]],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': False,
  'manual_cloud_cer': 0.0,
  'manual_cloud_cwp': 0.0,
  'manual_cloud_cwp_expr': '0.0',
  'manual_cloud_cth': 0.0,
  'manual_cloud_cbh': 0.0,
  'manual_cloud_cot': 0.0,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 5, 28),\n'
                   '                        tmhr_ranges_select=[[15.610, 15.822],\n'
                   '                                            [16.905, 17.404] \n'
                   '                                            ],\n'
                   "                        case_tag='clear_atm_corr',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=0.5,\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_lrt=atm_corr_overwrite_lrt,\n'
                   '                        manual_cloud=False,\n'
                   '                        manual_cloud_cer=0.0,\n'
                   '                        manual_cloud_cwp=0.0,\n'
                   '                        manual_cloud_cth=0.0,\n'
                   '                        manual_cloud_cbh=0.0,\n'
                   '                        manual_cloud_cot=0.0,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_030',
  'date': '2024-05-31',
  'case_tag': 'clear_atm_corr',
  'tmhr_ranges_select': [[13.839, 15.18]],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': False,
  'manual_cloud_cer': 0.0,
  'manual_cloud_cwp': 0.0,
  'manual_cloud_cwp_expr': '0.0',
  'manual_cloud_cth': 0.0,
  'manual_cloud_cbh': 0.0,
  'manual_cloud_cot': 0.0,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 5, 31),\n'
                   '                        tmhr_ranges_select=[[13.839, 15.180],  # 5.6 km\n'
                   '                                            ],\n'
                   "                        case_tag='clear_atm_corr',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=0.5,\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_lrt=atm_corr_overwrite_lrt,\n'
                   '                        manual_cloud=False,\n'
                   '                        manual_cloud_cer=0.0,\n'
                   '                        manual_cloud_cwp=0.0,\n'
                   '                        manual_cloud_cth=0.0,\n'
                   '                        manual_cloud_cbh=0.0,\n'
                   '                        manual_cloud_cot=0.0,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_031',
  'date': '2024-05-31',
  'case_tag': 'clear_atm_corr_2',
  'tmhr_ranges_select': [[16.905, 17.404]],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': False,
  'manual_cloud_cer': 0.0,
  'manual_cloud_cwp': 0.0,
  'manual_cloud_cwp_expr': '0.0',
  'manual_cloud_cth': 0.0,
  'manual_cloud_cbh': 0.0,
  'manual_cloud_cot': 0.0,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 5, 31),\n'
                   '                        tmhr_ranges_select=[\n'
                   '                                            [16.905, 17.404] \n'
                   '                                            ],\n'
                   "                        case_tag='clear_atm_corr_2',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=0.5,\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_lrt=atm_corr_overwrite_lrt,\n'
                   '                        manual_cloud=False,\n'
                   '                        manual_cloud_cer=0.0,\n'
                   '                        manual_cloud_cwp=0.0,\n'
                   '                        manual_cloud_cth=0.0,\n'
                   '                        manual_cloud_cbh=0.0,\n'
                   '                        manual_cloud_cot=0.0,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_032',
  'date': '2024-06-03',
  'case_tag': 'cloudy_atm_corr_1',
  'tmhr_ranges_select': [[13.62, 13.75]],
  'simulation_interval': 0.5,
  'clear_sky': False,
  'manual_cloud': True,
  'has_custom_levels': True,
  'manual_cloud_cer': 13.0,
  'manual_cloud_cwp': 0.07781999999999999,
  'manual_cloud_cwp_expr': '77.82 / 1000',
  'manual_cloud_cth': 1.93,
  'manual_cloud_cbh': 1.41,
  'manual_cloud_cot': 21.27,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 6, 3),\n'
                   '                        tmhr_ranges_select=[[13.62, 13.75],  # 300m, cloudy, camera icing\n'
                   '                                            ],\n'
                   "                        case_tag='cloudy_atm_corr_1',\n"
                   '                        config=config,\n'
                   '                        levels=np.concatenate((np.array([0.0, 0.2, 0.3, 0.4, 0.7, 1.0,]),\n'
                   '                                               np.array([1.41, 1.5, 1.93, 2.0, 2.5, 3.0, 4.0]), \n'
                   '                                               np.arange(5.0, 10.1, 2.5),\n'
                   '                                               np.array([15, 20, 30., 40., 45.]))),\n'
                   '                        simulation_interval=0.5,\n'
                   '                        clear_sky=False,\n'
                   '                        overwrite_lrt=atm_corr_overwrite_lrt,\n'
                   '                        manual_cloud=True,\n'
                   '                        manual_cloud_cer=13.0 ,\n'
                   '                        manual_cloud_cwp=77.82/1000,\n'
                   '                        manual_cloud_cth=1.93,\n'
                   '                        manual_cloud_cbh=1.41,\n'
                   '                        manual_cloud_cot=21.27,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_033',
  'date': '2024-06-03',
  'case_tag': 'cloudy_atm_corr_2',
  'tmhr_ranges_select': [[14.711, 14.868]],
  'simulation_interval': 0.5,
  'clear_sky': False,
  'manual_cloud': True,
  'has_custom_levels': True,
  'manual_cloud_cer': 7.0,
  'manual_cloud_cwp': 0.11365,
  'manual_cloud_cwp_expr': '113.65 / 1000',
  'manual_cloud_cth': 1.91,
  'manual_cloud_cbh': 0.5,
  'manual_cloud_cot': 24.31,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 6, 3),\n'
                   '                        tmhr_ranges_select=[[14.711, 14.868],  # 300m, cloudy, camera icing\n'
                   '                                            ],\n'
                   "                        case_tag='cloudy_atm_corr_2',\n"
                   '                        config=config,\n'
                   '                        levels=np.concatenate((np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, '
                   '1.0,]),\n'
                   '                                               np.array([1.5, 1.91, 2.0, 2.5, 3.0, 4.0]), \n'
                   '                                               np.arange(5.0, 10.1, 2.5),\n'
                   '                                               np.array([15, 20, 30., 40., 45.]))),\n'
                   '                        simulation_interval=0.5,\n'
                   '                        clear_sky=False,\n'
                   '                        overwrite_lrt=atm_corr_overwrite_lrt,\n'
                   '                        manual_cloud=True,\n'
                   '                        manual_cloud_cer=7.0,\n'
                   '                        manual_cloud_cwp=113.65/1000,\n'
                   '                        manual_cloud_cth=1.91,\n'
                   '                        manual_cloud_cbh=0.50,\n'
                   '                        manual_cloud_cot=24.31,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_034',
  'date': '2024-06-05',
  'case_tag': 'clear_atm_corr_1',
  'tmhr_ranges_select': [[12.405, 13.812]],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': False,
  'manual_cloud_cer': 0.0,
  'manual_cloud_cwp': 0.0,
  'manual_cloud_cwp_expr': '0.0',
  'manual_cloud_cth': 0.0,
  'manual_cloud_cbh': 0.0,
  'manual_cloud_cot': 0.0,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 6, 5),\n'
                   '                        tmhr_ranges_select=[[12.405, 13.812], # 5.7m,\n'
                   '                                            ],\n'
                   "                        case_tag='clear_atm_corr_1',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=0.5,\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_lrt=atm_corr_overwrite_lrt,\n'
                   '                        manual_cloud=False,\n'
                   '                        manual_cloud_cer=0.0,\n'
                   '                        manual_cloud_cwp=0.0,\n'
                   '                        manual_cloud_cth=0.0,\n'
                   '                        manual_cloud_cbh=0.0,\n'
                   '                        manual_cloud_cot=0.0,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_035',
  'date': '2024-06-05',
  'case_tag': 'clear_atm_corr_2',
  'tmhr_ranges_select': [[14.258, 15.036]],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': False,
  'manual_cloud_cer': 0.0,
  'manual_cloud_cwp': 0.0,
  'manual_cloud_cwp_expr': '0.0',
  'manual_cloud_cth': 0.0,
  'manual_cloud_cbh': 0.0,
  'manual_cloud_cot': 0.0,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 6, 5),\n'
                   '                        tmhr_ranges_select=[\n'
                   '                                            [14.258, 15.036], # 100m\n'
                   '                                            ],\n'
                   "                        case_tag='clear_atm_corr_2',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=0.5,\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_lrt=atm_corr_overwrite_lrt,\n'
                   '                        manual_cloud=False,\n'
                   '                        manual_cloud_cer=0.0,\n'
                   '                        manual_cloud_cwp=0.0,\n'
                   '                        manual_cloud_cth=0.0,\n'
                   '                        manual_cloud_cbh=0.0,\n'
                   '                        manual_cloud_cot=0.0,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_036',
  'date': '2024-06-05',
  'case_tag': 'clear_atm_corr_3',
  'tmhr_ranges_select': [[15.535, 15.931]],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': False,
  'manual_cloud_cer': 0.0,
  'manual_cloud_cwp': 0.0,
  'manual_cloud_cwp_expr': '0.0',
  'manual_cloud_cth': 0.0,
  'manual_cloud_cbh': 0.0,
  'manual_cloud_cot': 0.0,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 6, 5),\n'
                   '                        tmhr_ranges_select=[\n'
                   '                                            [15.535, 15.931], # 450m\n'
                   '                                            ],\n'
                   "                        case_tag='clear_atm_corr_3',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=0.5,\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_lrt=atm_corr_overwrite_lrt,\n'
                   '                        manual_cloud=False,\n'
                   '                        manual_cloud_cer=0.0,\n'
                   '                        manual_cloud_cwp=0.0,\n'
                   '                        manual_cloud_cth=0.0,\n'
                   '                        manual_cloud_cbh=0.0,\n'
                   '                        manual_cloud_cot=0.0,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_037',
  'date': '2024-06-05',
  'case_tag': 'clear_sky_spiral_atm_corr',
  'tmhr_ranges_select': [[13.7889, 13.801],
                         [13.835, 13.8395],
                         [13.878, 13.8885],
                         [13.924, 13.9255],
                         [13.9389, 13.9403],
                         [13.954, 13.9715],
                         [13.998, 14.0153],
                         [14.0417, 14.0575],
                         [14.0417, 14.0475],
                         [14.056, 14.059],
                         [14.0825, 14.0975],
                         [14.1264, 14.1525],
                         [14.1762, 14.1975],
                         [14.2194, 14.242],
                         [14.2605, 14.281]],
  'simulation_interval': None,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': False,
  'manual_cloud_cer': 0.0,
  'manual_cloud_cwp': 0.0,
  'manual_cloud_cwp_expr': '0.0',
  'manual_cloud_cth': 0.0,
  'manual_cloud_cbh': 0.0,
  'manual_cloud_cot': 0.0,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 6, 5),\n'
                   '                        tmhr_ranges_select=[\n'
                   '                                            [13.7889, 13.8010],\n'
                   '                                            [13.8350, 13.8395],\n'
                   '                                            [13.8780, 13.8885],\n'
                   '                                            [13.9240, 13.9255],\n'
                   '                                            [13.9389, 13.9403],\n'
                   '                                            [13.9540, 13.9715],\n'
                   '                                            [13.9980, 14.0153],\n'
                   '                                            [14.0417, 14.0575],\n'
                   '                                            [14.0417, 14.0475],\n'
                   '                                            [14.0560, 14.0590],\n'
                   '                                            [14.0825, 14.0975],\n'
                   '                                            [14.1264, 14.1525],\n'
                   '                                            [14.1762, 14.1975],\n'
                   '                                            [14.2194, 14.2420],\n'
                   '                                            [14.2605, 14.2810]\n'
                   '                                            ],\n'
                   "                        case_tag='clear_sky_spiral_atm_corr',\n"
                   '                        config=config,\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_lrt=atm_corr_overwrite_lrt,\n'
                   '                        manual_cloud=False,\n'
                   '                        manual_cloud_cer=0.0,\n'
                   '                        manual_cloud_cwp=0.0,\n'
                   '                        manual_cloud_cth=0.0,\n'
                   '                        manual_cloud_cbh=0.0,\n'
                   '                        manual_cloud_cot=0.0,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_038',
  'date': '2024-06-06',
  'case_tag': 'clear_atm_corr',
  'tmhr_ranges_select': [[16.25, 16.325], [16.375, 16.632], [16.7, 16.794], [16.85, 16.952]],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': False,
  'manual_cloud_cer': 0.0,
  'manual_cloud_cwp': 0.0,
  'manual_cloud_cwp_expr': '0.0',
  'manual_cloud_cth': 0.0,
  'manual_cloud_cbh': 0.0,
  'manual_cloud_cot': 0.0,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 6, 6),\n'
                   '                        tmhr_ranges_select=[[16.250, 16.325], # 100m, \n'
                   '                                            [16.375, 16.632], # 450m\n'
                   '                                            [16.700, 16.794], # 100m\n'
                   '                                            [16.850, 16.952], # 1.2km\n'
                   '                                            ],\n'
                   "                        case_tag='clear_atm_corr',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=0.5,\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_lrt=atm_corr_overwrite_lrt,\n'
                   '                        manual_cloud=False,\n'
                   '                        manual_cloud_cer=0.0,\n'
                   '                        manual_cloud_cwp=0.0,\n'
                   '                        manual_cloud_cth=0.0,\n'
                   '                        manual_cloud_cbh=0.0,\n'
                   '                        manual_cloud_cot=0.0,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_039',
  'date': '2024-06-07',
  'case_tag': 'cloudy_atm_corr',
  'tmhr_ranges_select': [[15.319, 15.763]],
  'simulation_interval': 0.5,
  'clear_sky': False,
  'manual_cloud': True,
  'has_custom_levels': True,
  'manual_cloud_cer': 6.7,
  'manual_cloud_cwp': 0.02696,
  'manual_cloud_cwp_expr': '26.96 / 1000',
  'manual_cloud_cth': 0.43,
  'manual_cloud_cbh': 0.15,
  'manual_cloud_cot': 6.02,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 6, 7),\n'
                   '                        tmhr_ranges_select=[[15.319, 15.763], # 100m, cloudy\n'
                   '                                            ],\n'
                   "                        case_tag='cloudy_atm_corr',\n"
                   '                        config=config,\n'
                   '                        levels=np.concatenate((np.array([0.0, 0.1, 0.15, 0.2, 0.43, 0.5, 0.6, 0.8, '
                   '1.0,]),\n'
                   '                                               np.array([1.5, 2.0, 2.5, 3.0, 4.0]), \n'
                   '                                               np.arange(5.0, 10.1, 2.5),\n'
                   '                                               np.array([15, 20, 30., 40., 45.]))),\n'
                   '                        simulation_interval=0.5,\n'
                   '                        clear_sky=False,\n'
                   '                        overwrite_lrt=atm_corr_overwrite_lrt,\n'
                   '                        manual_cloud=True,\n'
                   '                        manual_cloud_cer=6.7,\n'
                   '                        manual_cloud_cwp=26.96/1000,\n'
                   '                        manual_cloud_cth=0.43,\n'
                   '                        manual_cloud_cbh=0.15,\n'
                   '                        manual_cloud_cot=6.02,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_040',
  'date': '2024-06-11',
  'case_tag': 'clear_sky_spiral_atm_corr',
  'tmhr_ranges_select': [[14.5667, 14.5694],
                         [14.5986, 14.6097],
                         [14.6375, 14.6486],
                         [14.6778, 14.6903],
                         [14.7208, 14.7403],
                         [14.7653, 14.7875],
                         [14.8125, 14.8278],
                         [14.8542, 14.8736],
                         [14.8986, 14.9389]],
  'simulation_interval': None,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': False,
  'manual_cloud_cer': 0.0,
  'manual_cloud_cwp': 0.0,
  'manual_cloud_cwp_expr': '0.0',
  'manual_cloud_cth': 0.0,
  'manual_cloud_cbh': 0.0,
  'manual_cloud_cot': 0.0,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 6, 11),\n'
                   '                        tmhr_ranges_select=[[14.5667, 14.5694],\n'
                   '                                            [14.5986, 14.6097],\n'
                   '                                            [14.6375, 14.6486], # cloud shadow\n'
                   '                                            [14.6778, 14.6903],\n'
                   '                                            [14.7208, 14.7403],\n'
                   '                                            [14.7653, 14.7875],\n'
                   '                                            [14.8125, 14.8278],\n'
                   '                                            [14.8542, 14.8736],\n'
                   '                                            [14.8986, 14.9389], # more cracks\n'
                   '                                            ],\n'
                   "                        case_tag='clear_sky_spiral_atm_corr',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=None,\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_lrt=atm_corr_overwrite_lrt,\n'
                   '                        manual_cloud=False,\n'
                   '                        manual_cloud_cer=0.0,\n'
                   '                        manual_cloud_cwp=0.0,\n'
                   '                        manual_cloud_cth=0.0,\n'
                   '                        manual_cloud_cbh=0.0,\n'
                   '                        manual_cloud_cot=0.0,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_041',
  'date': '2024-06-11',
  'case_tag': 'clear_atm_corr_1',
  'tmhr_ranges_select': [[14.968, 15.229], [14.968, 15.347]],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': False,
  'manual_cloud_cer': 0.0,
  'manual_cloud_cwp': 0.0,
  'manual_cloud_cwp_expr': '0.0',
  'manual_cloud_cth': 0.0,
  'manual_cloud_cbh': 0.0,
  'manual_cloud_cot': 0.0,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 6, 11),\n'
                   '                        tmhr_ranges_select=[\n'
                   '                                            [14.968, 15.229], # 100, clear, some cloud\n'
                   '                                            [14.968, 15.347],\n'
                   '                                            ],\n'
                   "                        case_tag='clear_atm_corr_1',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=0.5,\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_lrt=atm_corr_overwrite_lrt,\n'
                   '                        manual_cloud=False,\n'
                   '                        manual_cloud_cer=0.0,\n'
                   '                        manual_cloud_cwp=0.0,\n'
                   '                        manual_cloud_cth=0.0,\n'
                   '                        manual_cloud_cbh=0.0,\n'
                   '                        manual_cloud_cot=0.0,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_042',
  'date': '2024-06-11',
  'case_tag': 'clear_atm_corr_2',
  'tmhr_ranges_select': [[15.347, 15.813], [15.813, 16.115]],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': False,
  'manual_cloud_cer': 0.0,
  'manual_cloud_cwp': 0.0,
  'manual_cloud_cwp_expr': '0.0',
  'manual_cloud_cth': 0.0,
  'manual_cloud_cbh': 0.0,
  'manual_cloud_cot': 0.0,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 6, 11),\n'
                   '                        tmhr_ranges_select=[\n'
                   '                                            [15.347, 15.813], # 100m\n'
                   '                                            [15.813, 16.115], # 100-450m, clear, some cloud\n'
                   '                                            ],\n'
                   "                        case_tag='clear_atm_corr_2',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=0.5,\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_lrt=atm_corr_overwrite_lrt,\n'
                   '                        manual_cloud=False,\n'
                   '                        manual_cloud_cer=0.0,\n'
                   '                        manual_cloud_cwp=0.0,\n'
                   '                        manual_cloud_cth=0.0,\n'
                   '                        manual_cloud_cbh=0.0,\n'
                   '                        manual_cloud_cot=0.0,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_043',
  'date': '2024-06-13',
  'case_tag': 'clear_atm_corr_1',
  'tmhr_ranges_select': [[13.704, 13.817]],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': False,
  'manual_cloud_cer': 0.0,
  'manual_cloud_cwp': 0.0,
  'manual_cloud_cwp_expr': '0.0',
  'manual_cloud_cth': 0.0,
  'manual_cloud_cbh': 0.0,
  'manual_cloud_cot': 0.0,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 6, 13),\n'
                   '                        tmhr_ranges_select=[[13.704, 13.817], # 100-450m, clear, some cloud\n'
                   '                                            ],\n'
                   "                        case_tag='clear_atm_corr_1',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=0.5,\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_lrt=atm_corr_overwrite_lrt,\n'
                   '                        manual_cloud=False,\n'
                   '                        manual_cloud_cer=0.0,\n'
                   '                        manual_cloud_cwp=0.0,\n'
                   '                        manual_cloud_cth=0.0,\n'
                   '                        manual_cloud_cbh=0.0,\n'
                   '                        manual_cloud_cot=0.0,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_044',
  'date': '2024-06-13',
  'case_tag': 'cloudy_atm_corr_1',
  'tmhr_ranges_select': [[14.109, 14.14]],
  'simulation_interval': 0.5,
  'clear_sky': False,
  'manual_cloud': True,
  'has_custom_levels': True,
  'manual_cloud_cer': 17.4,
  'manual_cloud_cwp': 0.09051000000000001,
  'manual_cloud_cwp_expr': '90.51 / 1000',
  'manual_cloud_cth': 0.52,
  'manual_cloud_cbh': 0.15,
  'manual_cloud_cot': 7.82,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 6, 13),\n'
                   '                        tmhr_ranges_select=[[14.109, 14.140], # 100m, cloudy\n'
                   '                                            ],\n'
                   "                        case_tag='cloudy_atm_corr_1',\n"
                   '                        config=config,\n'
                   '                        levels=np.concatenate((np.array([0.0, 0.1, 0.15, 0.2, 0.4, 0.52, 0.6, 0.8, '
                   '1.0,]),\n'
                   '                                               np.array([1.5, 2.0, 2.5, 3.0, 4.0]), \n'
                   '                                               np.arange(5.0, 10.1, 2.5),\n'
                   '                                               np.array([15, 20, 30., 40., 45.]))),\n'
                   '                        simulation_interval=0.5,\n'
                   '                        clear_sky=False,\n'
                   '                        overwrite_lrt=atm_corr_overwrite_lrt,\n'
                   '                        manual_cloud=True,\n'
                   '                        manual_cloud_cer=17.4,\n'
                   '                        manual_cloud_cwp=90.51/1000,\n'
                   '                        manual_cloud_cth=0.52,\n'
                   '                        manual_cloud_cbh=0.15,\n'
                   '                        manual_cloud_cot=7.82,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_045',
  'date': '2024-06-13',
  'case_tag': 'cloudy_atm_corr_2',
  'tmhr_ranges_select': [[15.834, 15.883]],
  'simulation_interval': 0.5,
  'clear_sky': False,
  'manual_cloud': True,
  'has_custom_levels': True,
  'manual_cloud_cer': 22.4,
  'manual_cloud_cwp': 0.0356,
  'manual_cloud_cwp_expr': '35.6 / 1000',
  'manual_cloud_cth': 0.58,
  'manual_cloud_cbh': 0.28,
  'manual_cloud_cot': 2.39,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 6, 13),\n'
                   '                        tmhr_ranges_select=[[15.834, 15.883], # 100m, cloudy\n'
                   '                                            ],\n'
                   "                        case_tag='cloudy_atm_corr_2',\n"
                   '                        config=config,\n'
                   '                        levels=np.concatenate((np.array([0.0, 0.1, 0.2, 0.28, 0.3, 0.5, 0.58, 0.8, '
                   '1.0,]),\n'
                   '                                               np.array([1.5, 2.0, 2.5, 3.0, 4.0]), \n'
                   '                                               np.arange(5.0, 10.1, 2.5),\n'
                   '                                               np.array([15, 20, 30., 40., 45.]))),\n'
                   '                        simulation_interval=0.5,\n'
                   '                        clear_sky=False,\n'
                   '                        overwrite_lrt=atm_corr_overwrite_lrt,\n'
                   '                        manual_cloud=True,\n'
                   '                        manual_cloud_cer=22.4,\n'
                   '                        manual_cloud_cwp=35.6/1000,\n'
                   '                        manual_cloud_cth=0.58,\n'
                   '                        manual_cloud_cbh=0.28,\n'
                   '                        manual_cloud_cot=2.39,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_046',
  'date': '2024-06-13',
  'case_tag': 'cloudy_atm_corr_3',
  'tmhr_ranges_select': [[16.043, 16.067]],
  'simulation_interval': 0.5,
  'clear_sky': False,
  'manual_cloud': True,
  'has_custom_levels': True,
  'manual_cloud_cer': 8.9,
  'manual_cloud_cwp': 0.02129,
  'manual_cloud_cwp_expr': '21.29 / 1000',
  'manual_cloud_cth': 0.68,
  'manual_cloud_cbh': 0.38,
  'manual_cloud_cot': 3.59,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 6, 13),\n'
                   '                        tmhr_ranges_select=[[16.043, 16.067], # 100-200m, cloudy\n'
                   '                                            ],\n'
                   "                        case_tag='cloudy_atm_corr_3',\n"
                   '                        config=config,\n'
                   '                        levels=np.concatenate((np.array([0.0, 0.1, 0.2, 0.3, 0.38, 0.5, 0.68, 0.8, '
                   '1.0,]),\n'
                   '                                               np.array([1.5, 2.0, 2.5, 3.0, 4.0]), \n'
                   '                                               np.arange(5.0, 10.1, 2.5),\n'
                   '                                               np.array([15, 20, 30., 40., 45.]))),\n'
                   '                        simulation_interval=0.5,\n'
                   '                        clear_sky=False,\n'
                   '                        overwrite_lrt=atm_corr_overwrite_lrt,\n'
                   '                        manual_cloud=True,\n'
                   '                        manual_cloud_cer=8.9,\n'
                   '                        manual_cloud_cwp=21.29/1000,\n'
                   '                        manual_cloud_cth=0.68,\n'
                   '                        manual_cloud_cbh=0.38,\n'
                   '                        manual_cloud_cot=3.59,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_047',
  'date': '2024-06-13',
  'case_tag': 'clear_atm_corr_2',
  'tmhr_ranges_select': [[16.55, 17.581]],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': False,
  'manual_cloud_cer': 0.0,
  'manual_cloud_cwp': 0.0,
  'manual_cloud_cwp_expr': '0.0',
  'manual_cloud_cth': 0.0,
  'manual_cloud_cbh': 0.0,
  'manual_cloud_cot': 0.0,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 6, 13),\n'
                   '                        tmhr_ranges_select=[[16.550, 17.581], # 100-500m, clear\n'
                   '                                            ],\n'
                   "                        case_tag='clear_atm_corr_2',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=0.5,\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_lrt=atm_corr_overwrite_lrt,\n'
                   '                        manual_cloud=False,\n'
                   '                        manual_cloud_cer=0.0,\n'
                   '                        manual_cloud_cwp=0.0,\n'
                   '                        manual_cloud_cth=0.0,\n'
                   '                        manual_cloud_cbh=0.0,\n'
                   '                        manual_cloud_cot=0.0,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_048',
  'date': '2024-07-25',
  'case_tag': 'cloudy_atm_corr',
  'tmhr_ranges_select': [[15.094, 15.3]],
  'simulation_interval': 0.5,
  'clear_sky': False,
  'manual_cloud': True,
  'has_custom_levels': True,
  'manual_cloud_cer': 11.4,
  'manual_cloud_cwp': 0.00994,
  'manual_cloud_cwp_expr': '9.94 / 1000',
  'manual_cloud_cth': 0.3,
  'manual_cloud_cbh': 0.16,
  'manual_cloud_cot': 1.31,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 7, 25),\n'
                   '                        tmhr_ranges_select=[[15.094, 15.300], # 100m, some low clouds or fog '
                   'below\n'
                   '                                            ],\n'
                   "                        case_tag='cloudy_atm_corr',\n"
                   '                        config=config,\n'
                   '                        levels=np.concatenate((np.array([0.0, 0.1, 0.16, 0.2, 0.3, 0.4, 0.6, 0.8, '
                   '1.0,]),\n'
                   '                                               np.array([1.5, 2.0, 2.5, 3.0, 4.0]), \n'
                   '                                               np.arange(5.0, 10.1, 2.5),\n'
                   '                                               np.array([15, 20, 30., 40., 45.]))),\n'
                   '                        simulation_interval=0.5,\n'
                   '                        clear_sky=False,\n'
                   '                        overwrite_lrt=atm_corr_overwrite_lrt,\n'
                   '                        manual_cloud=True,\n'
                   '                        manual_cloud_cer=11.4,\n'
                   '                        manual_cloud_cwp=9.94/1000,\n'
                   '                        manual_cloud_cth=0.30,\n'
                   '                        manual_cloud_cbh=0.16,\n'
                   '                        manual_cloud_cot=1.31,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_049',
  'date': '2024-07-25',
  'case_tag': 'cloudy_atm_corr_2',
  'tmhr_ranges_select': [[15.881, 15.903]],
  'simulation_interval': 0.5,
  'clear_sky': False,
  'manual_cloud': True,
  'has_custom_levels': True,
  'manual_cloud_cer': 11.4,
  'manual_cloud_cwp': 0.00994,
  'manual_cloud_cwp_expr': '9.94 / 1000',
  'manual_cloud_cth': 0.3,
  'manual_cloud_cbh': 0.16,
  'manual_cloud_cot': 1.31,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 7, 25),\n'
                   '                        tmhr_ranges_select=[[15.881, 15.903], # 200-500m\n'
                   '                                            ],\n'
                   "                        case_tag='cloudy_atm_corr_2',\n"
                   '                        config=config,\n'
                   '                        levels=np.concatenate((np.array([0.0, 0.1, 0.16, 0.2, 0.3, 0.4, 0.6, 0.8, '
                   '1.0,]),\n'
                   '                                               np.array([1.5, 2.0, 2.5, 3.0, 4.0]), \n'
                   '                                               np.arange(5.0, 10.1, 2.5),\n'
                   '                                               np.array([15, 20, 30., 40., 45.]))),\n'
                   '                        simulation_interval=0.5,\n'
                   '                        clear_sky=False,\n'
                   '                        overwrite_lrt=atm_corr_overwrite_lrt,\n'
                   '                        manual_cloud=True,\n'
                   '                        manual_cloud_cer=11.4,\n'
                   '                        manual_cloud_cwp=9.94/1000,\n'
                   '                        manual_cloud_cth=0.30,\n'
                   '                        manual_cloud_cbh=0.16,\n'
                   '                        manual_cloud_cot=1.31,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_050',
  'date': '2024-07-29',
  'case_tag': 'clear_sky_spiral_atm_corr',
  'tmhr_ranges_select': [[13.442, 13.465],
                         [13.49, 13.514],
                         [13.536, 13.554],
                         [13.58, 13.611],
                         [13.639, 13.654],
                         [13.676, 13.707],
                         [13.733, 13.775],
                         [13.793, 13.836]],
  'simulation_interval': None,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': False,
  'manual_cloud_cer': 0.0,
  'manual_cloud_cwp': 0.0,
  'manual_cloud_cwp_expr': '0.0',
  'manual_cloud_cth': 0.0,
  'manual_cloud_cbh': 0.0,
  'manual_cloud_cot': 0.0,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 7, 29),\n'
                   '                        tmhr_ranges_select=[[13.442, 13.465],\n'
                   '                                            [13.490, 13.514],\n'
                   '                                            [13.536, 13.554],\n'
                   '                                            [13.580, 13.611],\n'
                   '                                            [13.639, 13.654],\n'
                   '                                            [13.676, 13.707],\n'
                   '                                            [13.733, 13.775],\n'
                   '                                            [13.793, 13.836],\n'
                   '                                            ],\n'
                   "                        case_tag='clear_sky_spiral_atm_corr',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=None,\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_lrt=atm_corr_overwrite_lrt,\n'
                   '                        manual_cloud=False,\n'
                   '                        manual_cloud_cer=0.0,\n'
                   '                        manual_cloud_cwp=0.0,\n'
                   '                        manual_cloud_cth=0.0,\n'
                   '                        manual_cloud_cbh=0.0,\n'
                   '                        manual_cloud_cot=0.0,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_051',
  'date': '2024-07-29',
  'case_tag': 'clear_atm_corr_1',
  'tmhr_ranges_select': [[13.939, 14.2], [14.438, 14.714]],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': False,
  'manual_cloud_cer': 0.0,
  'manual_cloud_cwp': 0.0,
  'manual_cloud_cwp_expr': '0.0',
  'manual_cloud_cth': 0.0,
  'manual_cloud_cbh': 0.0,
  'manual_cloud_cot': 0.0,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 7, 29),\n'
                   '                        tmhr_ranges_select=[[13.939, 14.200], # 100m, clear\n'
                   '                                            [14.438, 14.714], # 3.7km\n'
                   '                                            ],\n'
                   "                        case_tag='clear_atm_corr_1',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=0.5,\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_lrt=atm_corr_overwrite_lrt,\n'
                   '                        manual_cloud=False,\n'
                   '                        manual_cloud_cer=0.0,\n'
                   '                        manual_cloud_cwp=0.0,\n'
                   '                        manual_cloud_cth=0.0,\n'
                   '                        manual_cloud_cbh=0.0,\n'
                   '                        manual_cloud_cot=0.0,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_052',
  'date': '2024-07-29',
  'case_tag': 'clear_atm_corr_2',
  'tmhr_ranges_select': [[15.214, 15.804], [16.176, 16.304]],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': False,
  'manual_cloud_cer': 0.0,
  'manual_cloud_cwp': 0.0,
  'manual_cloud_cwp_expr': '0.0',
  'manual_cloud_cth': 0.0,
  'manual_cloud_cbh': 0.0,
  'manual_cloud_cot': 0.0,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 7, 29),\n'
                   '                        tmhr_ranges_select=[\n'
                   '                                            [15.214, 15.804], # 1.3km\n'
                   '                                            [16.176, 16.304], # 1.3km\n'
                   '                                            ],\n'
                   "                        case_tag='clear_atm_corr_2',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=0.5,\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_lrt=atm_corr_overwrite_lrt,\n'
                   '                        manual_cloud=False,\n'
                   '                        manual_cloud_cer=0.0,\n'
                   '                        manual_cloud_cwp=0.0,\n'
                   '                        manual_cloud_cth=0.0,\n'
                   '                        manual_cloud_cbh=0.0,\n'
                   '                        manual_cloud_cot=0.0,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_053',
  'date': '2024-07-30',
  'case_tag': 'clear_sky_spiral_atm_corr',
  'tmhr_ranges_select': [[13.886, 13.908],
                         [13.934, 13.95],
                         [13.976, 14.0],
                         [14.031, 14.051],
                         [14.073, 14.096],
                         [14.115, 14.134],
                         [14.157, 14.179],
                         [14.202, 14.219],
                         [14.239, 14.254],
                         [14.275, 14.294]],
  'simulation_interval': None,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': False,
  'manual_cloud_cer': 0.0,
  'manual_cloud_cwp': 0.0,
  'manual_cloud_cwp_expr': '0.0',
  'manual_cloud_cth': 0.0,
  'manual_cloud_cbh': 0.0,
  'manual_cloud_cot': 0.0,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 7, 30),\n'
                   '                        tmhr_ranges_select=[[13.886, 13.908],\n'
                   '                                            [13.934, 13.950],\n'
                   '                                            [13.976, 14.000],\n'
                   '                                            [14.031, 14.051],\n'
                   '                                            [14.073, 14.096],\n'
                   '                                            [14.115, 14.134],\n'
                   '                                            [14.157, 14.179],\n'
                   '                                            [14.202, 14.219],\n'
                   '                                            [14.239, 14.254],\n'
                   '                                            [14.275, 14.294],\n'
                   '                                            ],\n'
                   "                        case_tag='clear_sky_spiral_atm_corr',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=None,\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_lrt=atm_corr_overwrite_lrt,\n'
                   '                        manual_cloud=False,\n'
                   '                        manual_cloud_cer=0.0,\n'
                   '                        manual_cloud_cwp=0.0,\n'
                   '                        manual_cloud_cth=0.0,\n'
                   '                        manual_cloud_cbh=0.0,\n'
                   '                        manual_cloud_cot=0.0,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_054',
  'date': '2024-07-30',
  'case_tag': 'clear_atm_corr',
  'tmhr_ranges_select': [[14.318, 14.936], [15.043, 15.14]],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': False,
  'manual_cloud_cer': 0.0,
  'manual_cloud_cwp': 0.0,
  'manual_cloud_cwp_expr': '0.0',
  'manual_cloud_cth': 0.0,
  'manual_cloud_cbh': 0.0,
  'manual_cloud_cot': 0.0,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 7, 30),\n'
                   '                        tmhr_ranges_select=[[14.318, 14.936], # 100-450m, clear\n'
                   '                                            [15.043, 15.140], # 1.5km\n'
                   '                                            ],\n'
                   "                        case_tag='clear_atm_corr',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=0.5,\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_lrt=atm_corr_overwrite_lrt,\n'
                   '                        manual_cloud=False,\n'
                   '                        manual_cloud_cer=0.0,\n'
                   '                        manual_cloud_cwp=0.0,\n'
                   '                        manual_cloud_cth=0.0,\n'
                   '                        manual_cloud_cbh=0.0,\n'
                   '                        manual_cloud_cot=0.0,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_055',
  'date': '2024-08-01',
  'case_tag': 'clear_atm_corr',
  'tmhr_ranges_select': [[13.843, 14.361]],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': False,
  'manual_cloud_cer': 0.0,
  'manual_cloud_cwp': 0.0,
  'manual_cloud_cwp_expr': '0.0',
  'manual_cloud_cth': 0.0,
  'manual_cloud_cbh': 0.0,
  'manual_cloud_cot': 0.0,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 8, 1),\n'
                   '                        tmhr_ranges_select=[[13.843, 14.361], # 100-450m, clear, some open ocean\n'
                   '                                            ],\n'
                   "                        case_tag='clear_atm_corr',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=0.5,\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_lrt=atm_corr_overwrite_lrt,\n'
                   '                        manual_cloud=False,\n'
                   '                        manual_cloud_cer=0.0,\n'
                   '                        manual_cloud_cwp=0.0,\n'
                   '                        manual_cloud_cth=0.0,\n'
                   '                        manual_cloud_cbh=0.0,\n'
                   '                        manual_cloud_cot=0.0,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_056',
  'date': '2024-08-01',
  'case_tag': 'clear_atm_corr_2',
  'tmhr_ranges_select': [[14.739, 15.053]],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': False,
  'manual_cloud_cer': 0.0,
  'manual_cloud_cwp': 0.0,
  'manual_cloud_cwp_expr': '0.0',
  'manual_cloud_cth': 0.0,
  'manual_cloud_cbh': 0.0,
  'manual_cloud_cot': 0.0,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 8, 1),\n'
                   '                        tmhr_ranges_select=[\n'
                   '                                            [14.739, 15.053], # 550m\n'
                   '                                            ],\n'
                   "                        case_tag='clear_atm_corr_2',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=0.5,\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_lrt=atm_corr_overwrite_lrt,\n'
                   '                        manual_cloud=False,\n'
                   '                        manual_cloud_cer=0.0,\n'
                   '                        manual_cloud_cwp=0.0,\n'
                   '                        manual_cloud_cth=0.0,\n'
                   '                        manual_cloud_cbh=0.0,\n'
                   '                        manual_cloud_cot=0.0,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_057',
  'date': '2024-08-02',
  'case_tag': 'clear_atm_corr_1',
  'tmhr_ranges_select': [[14.557, 15.1]],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': False,
  'manual_cloud_cer': 0.0,
  'manual_cloud_cwp': 0.0,
  'manual_cloud_cwp_expr': '0.0',
  'manual_cloud_cth': 0.0,
  'manual_cloud_cbh': 0.0,
  'manual_cloud_cot': 0.0,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 8, 2),\n'
                   '                        tmhr_ranges_select=[\n'
                   '                                            [14.557, 15.100], # 100m\n'
                   '                                            ],\n'
                   "                        case_tag='clear_atm_corr_1',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=0.5,\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_lrt=atm_corr_overwrite_lrt,\n'
                   '                        manual_cloud=False,\n'
                   '                        manual_cloud_cer=0.0,\n'
                   '                        manual_cloud_cwp=0.0,\n'
                   '                        manual_cloud_cth=0.0,\n'
                   '                        manual_cloud_cbh=0.0,\n'
                   '                        manual_cloud_cot=0.0,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_058',
  'date': '2024-08-02',
  'case_tag': 'clear_atm_corr_2',
  'tmhr_ranges_select': [[15.244, 16.635]],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': False,
  'manual_cloud_cer': 0.0,
  'manual_cloud_cwp': 0.0,
  'manual_cloud_cwp_expr': '0.0',
  'manual_cloud_cth': 0.0,
  'manual_cloud_cbh': 0.0,
  'manual_cloud_cot': 0.0,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 8, 2),\n'
                   '                        tmhr_ranges_select=[\n'
                   '                                            [15.244, 16.635], # 1km\n'
                   '                                            ],\n'
                   "                        case_tag='clear_atm_corr_2',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=0.5,\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_lrt=atm_corr_overwrite_lrt,\n'
                   '                        manual_cloud=False,\n'
                   '                        manual_cloud_cer=0.0,\n'
                   '                        manual_cloud_cwp=0.0,\n'
                   '                        manual_cloud_cth=0.0,\n'
                   '                        manual_cloud_cbh=0.0,\n'
                   '                        manual_cloud_cot=0.0,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_059',
  'date': '2024-08-07',
  'case_tag': 'clear_atm_corr_1',
  'tmhr_ranges_select': [[13.344, 13.763]],
  'simulation_interval': 0.5,
  'clear_sky': False,
  'manual_cloud': True,
  'has_custom_levels': True,
  'manual_cloud_cer': 10.7,
  'manual_cloud_cwp': 0.01128,
  'manual_cloud_cwp_expr': '11.28 / 1000',
  'manual_cloud_cth': 0.78,
  'manual_cloud_cbh': 0.69,
  'manual_cloud_cot': 1.59,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 8, 7),\n'
                   '                        tmhr_ranges_select=[[13.344, 13.763], # 100m, cloudy\n'
                   '                                            ],\n'
                   "                        case_tag='clear_atm_corr_1',\n"
                   '                        config=config,\n'
                   '                        levels=np.concatenate((np.array([0.0, 0.1, 0.2, 0.4, 0.6, 0.65, 0.69, '
                   '0.78, 1.0,]),\n'
                   '                                               np.array([1.5, 2.0, 2.5, 3.0, 4.0]), \n'
                   '                                               np.arange(5.0, 10.1, 2.5),\n'
                   '                                               np.array([15, 20, 30., 40., 45.]))),\n'
                   '                        simulation_interval=0.5,\n'
                   '                        clear_sky=False,\n'
                   '                        overwrite_lrt=atm_corr_overwrite_lrt,\n'
                   '                        manual_cloud=True,\n'
                   '                        manual_cloud_cer=10.7,\n'
                   '                        manual_cloud_cwp=11.28/1000,\n'
                   '                        manual_cloud_cth=0.78,\n'
                   '                        manual_cloud_cbh=0.69,\n'
                   '                        manual_cloud_cot=1.59,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_060',
  'date': '2024-08-07',
  'case_tag': 'cloudy_atm_corr_2',
  'tmhr_ranges_select': [[15.472, 15.567], [15.58, 15.921]],
  'simulation_interval': 0.5,
  'clear_sky': False,
  'manual_cloud': True,
  'has_custom_levels': True,
  'manual_cloud_cer': 7.2,
  'manual_cloud_cwp': 0.0775,
  'manual_cloud_cwp_expr': '77.5 / 1000',
  'manual_cloud_cth': 0.96,
  'manual_cloud_cbh': 0.62,
  'manual_cloud_cot': 16.21,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 8, 7),\n'
                   '                        tmhr_ranges_select=[\n'
                   '                                            [15.472, 15.567], # 180m, cloudy\n'
                   '                                            [15.580, 15.921], # 100m, cloudy\n'
                   '                                            ],\n'
                   "                        case_tag='cloudy_atm_corr_2',\n"
                   '                        config=config,\n'
                   '                        levels=np.concatenate((np.array([0.0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.62, 0.8, '
                   '0.96,]),\n'
                   '                                               np.array([1.5, 2.0, 2.5, 3.0, 4.0]), \n'
                   '                                               np.arange(5.0, 10.1, 2.5),\n'
                   '                                               np.array([15, 20, 30., 40., 45.]))),\n'
                   '                        simulation_interval=0.5,\n'
                   '                        clear_sky=False,\n'
                   '                        overwrite_lrt=atm_corr_overwrite_lrt,\n'
                   '                        manual_cloud=True,\n'
                   '                        manual_cloud_cer=7.2,\n'
                   '                        manual_cloud_cwp=77.5/1000,\n'
                   '                        manual_cloud_cth=0.96,\n'
                   '                        manual_cloud_cbh=0.62,\n'
                   '                        manual_cloud_cot=16.21,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_061',
  'date': '2024-08-08',
  'case_tag': 'clear_atm_corr_1',
  'tmhr_ranges_select': [[12.99, 13.18]],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': False,
  'manual_cloud_cer': 0.0,
  'manual_cloud_cwp': 0.0,
  'manual_cloud_cwp_expr': '0.0',
  'manual_cloud_cth': 0.0,
  'manual_cloud_cbh': 0.0,
  'manual_cloud_cot': 0.0,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 8, 8),\n'
                   '                        tmhr_ranges_select=[\n'
                   '                                            [12.990, 13.180], # 180m, clear\n'
                   '                                            ],\n'
                   "                        case_tag='clear_atm_corr_1',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=0.5,\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_lrt=atm_corr_overwrite_lrt,\n'
                   '                        manual_cloud=False,\n'
                   '                        manual_cloud_cer=0.0,\n'
                   '                        manual_cloud_cwp=0.0,\n'
                   '                        manual_cloud_cth=0.0,\n'
                   '                        manual_cloud_cbh=0.0,\n'
                   '                        manual_cloud_cot=0.0,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_062',
  'date': '2024-08-08',
  'case_tag': 'clear_atm_corr_2',
  'tmhr_ranges_select': [[14.25, 14.373]],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': False,
  'manual_cloud_cer': 0.0,
  'manual_cloud_cwp': 0.0,
  'manual_cloud_cwp_expr': '0.0',
  'manual_cloud_cth': 0.0,
  'manual_cloud_cbh': 0.0,
  'manual_cloud_cot': 0.0,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 8, 8),\n'
                   '                        tmhr_ranges_select=[\n'
                   '                                            [14.250, 14.373], # 180m, clear\n'
                   '                                            ],\n'
                   "                        case_tag='clear_atm_corr_2',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=0.5,\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_lrt=atm_corr_overwrite_lrt,\n'
                   '                        manual_cloud=False,\n'
                   '                        manual_cloud_cer=0.0,\n'
                   '                        manual_cloud_cwp=0.0,\n'
                   '                        manual_cloud_cth=0.0,\n'
                   '                        manual_cloud_cbh=0.0,\n'
                   '                        manual_cloud_cot=0.0,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_063',
  'date': '2024-08-08',
  'case_tag': 'clear_atm_corr_3',
  'tmhr_ranges_select': [[16.471, 16.601]],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': False,
  'manual_cloud_cer': 0.0,
  'manual_cloud_cwp': 0.0,
  'manual_cloud_cwp_expr': '0.0',
  'manual_cloud_cth': 0.0,
  'manual_cloud_cbh': 0.0,
  'manual_cloud_cot': 0.0,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 8, 8),\n'
                   '                        tmhr_ranges_select=[\n'
                   '                                            [16.471, 16.601], # 180m, clear\n'
                   '                                            ],\n'
                   "                        case_tag='clear_atm_corr_3',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=0.5,\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_lrt=atm_corr_overwrite_lrt,\n'
                   '                        manual_cloud=False,\n'
                   '                        manual_cloud_cer=0.0,\n'
                   '                        manual_cloud_cwp=0.0,\n'
                   '                        manual_cloud_cth=0.0,\n'
                   '                        manual_cloud_cbh=0.0,\n'
                   '                        manual_cloud_cot=0.0,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_064',
  'date': '2024-08-08',
  'case_tag': 'cloudy_atm_corr_1',
  'tmhr_ranges_select': [[13.212, 13.347]],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': True,
  'manual_cloud_cer': 15.3,
  'manual_cloud_cwp': 0.14393999999999998,
  'manual_cloud_cwp_expr': '143.94 / 1000',
  'manual_cloud_cth': 1.98,
  'manual_cloud_cbh': 0.67,
  'manual_cloud_cot': 14.12,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 8, 8),\n'
                   '                        tmhr_ranges_select=[\n'
                   '                                            [13.212, 13.347], # 100m, cloudy\n'
                   '                                            ],\n'
                   "                        case_tag='cloudy_atm_corr_1',\n"
                   '                        config=config,\n'
                   '                        levels=np.concatenate((np.array([0.0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.67, 0.8, '
                   '1.0,]),\n'
                   '                                               np.array([1.5, 1.98, 2.5, 3.0, 4.0]), \n'
                   '                                               np.arange(5.0, 10.1, 2.5),\n'
                   '                                               np.array([15, 20, 30., 40., 45.]))),\n'
                   '                        simulation_interval=0.5,\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_lrt=atm_corr_overwrite_lrt,\n'
                   '                        manual_cloud=False,\n'
                   '                        manual_cloud_cer=15.3,\n'
                   '                        manual_cloud_cwp=143.94/1000,\n'
                   '                        manual_cloud_cth=1.98,\n'
                   '                        manual_cloud_cbh=0.67,\n'
                   '                        manual_cloud_cot=14.12,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_065',
  'date': '2024-08-08',
  'case_tag': 'cloudy_atm_corr_2',
  'tmhr_ranges_select': [[15.314, 15.504]],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': True,
  'manual_cloud_cer': 7.8,
  'manual_cloud_cwp': 0.06418,
  'manual_cloud_cwp_expr': '64.18 / 1000',
  'manual_cloud_cth': 2.21,
  'manual_cloud_cbh': 1.81,
  'manual_cloud_cot': 12.41,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 8, 8),\n'
                   '                        tmhr_ranges_select=[\n'
                   '                                            [15.314, 15.504], # 100m, cloudy\n'
                   '                                            ],\n'
                   "                        case_tag='cloudy_atm_corr_2',\n"
                   '                        config=config,\n'
                   '                        levels=np.concatenate((np.array([0.0, 0.1, 0.2, 0.4, 0.78, 1.0,]),\n'
                   '                                               np.array([1.5, 1.81, 2.21, 2.5, 3.0, 4.0]), \n'
                   '                                               np.arange(5.0, 10.1, 2.5),\n'
                   '                                               np.array([15, 20, 30., 40., 45.]))),\n'
                   '                        simulation_interval=0.5,\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_lrt=atm_corr_overwrite_lrt,\n'
                   '                        manual_cloud=False,\n'
                   '                        manual_cloud_cer=7.8,\n'
                   '                        manual_cloud_cwp=64.18/1000,\n'
                   '                        manual_cloud_cth=2.21,\n'
                   '                        manual_cloud_cbh=1.81,\n'
                   '                        manual_cloud_cot=12.41,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_066',
  'date': '2024-08-09',
  'case_tag': 'cloudy_atm_corr_1',
  'tmhr_ranges_select': [[13.376, 13.6]],
  'simulation_interval': 0.5,
  'clear_sky': False,
  'manual_cloud': True,
  'has_custom_levels': True,
  'manual_cloud_cer': 9.0,
  'manual_cloud_cwp': 0.08349,
  'manual_cloud_cwp_expr': '83.49 / 1000',
  'manual_cloud_cth': 0.77,
  'manual_cloud_cbh': 0.34,
  'manual_cloud_cot': 13.93,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 8, 9),\n'
                   '                        tmhr_ranges_select=[\n'
                   '                                            [13.376, 13.600], # 100m, cloudy\n'
                   '                                            ],\n'
                   "                        case_tag='cloudy_atm_corr_1',\n"
                   '                        config=config,\n'
                   '                        levels=np.concatenate((np.array([0.0, 0.1, 0.15, 0.2, 0.34, 0.4, 0.6, '
                   '0.77, 1.0,]),\n'
                   '                                               np.array([1.5, 2.0, 2.5, 3.0, 4.0]), \n'
                   '                                               np.arange(5.0, 10.1, 2.5),\n'
                   '                                               np.array([15, 20, 30., 40., 45.]))),\n'
                   '                        simulation_interval=0.5,\n'
                   '                        clear_sky=False,\n'
                   '                        overwrite_lrt=atm_corr_overwrite_lrt,\n'
                   '                        manual_cloud=True,\n'
                   '                        manual_cloud_cer=9.0,\n'
                   '                        manual_cloud_cwp=83.49/1000,\n'
                   '                        manual_cloud_cth=0.77,\n'
                   '                        manual_cloud_cbh=0.34,\n'
                   '                        manual_cloud_cot=13.93,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_067',
  'date': '2024-08-09',
  'case_tag': 'clear_atm_corr',
  'tmhr_ranges_select': [[14.75, 15.06], [15.622, 15.887]],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': False,
  'manual_cloud_cer': 0.0,
  'manual_cloud_cwp': 0.0,
  'manual_cloud_cwp_expr': '0.0',
  'manual_cloud_cth': 0.0,
  'manual_cloud_cbh': 0.0,
  'manual_cloud_cot': 0.0,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 8, 9),\n'
                   '                        tmhr_ranges_select=[\n'
                   '                                            [14.750, 15.060], # 100m, clear\n'
                   '                                            [15.622, 15.887], # 100m, clear\n'
                   '                                            ],\n'
                   "                        case_tag='clear_atm_corr',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=0.5,\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_lrt=atm_corr_overwrite_lrt,\n'
                   '                        manual_cloud=False,\n'
                   '                        manual_cloud_cer=0.0,\n'
                   '                        manual_cloud_cwp=0.0,\n'
                   '                        manual_cloud_cth=0.0,\n'
                   '                        manual_cloud_cbh=0.0,\n'
                   '                        manual_cloud_cot=0.0,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_068',
  'date': '2024-08-09',
  'case_tag': 'cloudy_atm_corr_2',
  'tmhr_ranges_select': [[16.029, 16.224]],
  'simulation_interval': 0.5,
  'clear_sky': False,
  'manual_cloud': True,
  'has_custom_levels': True,
  'manual_cloud_cer': 8.3,
  'manual_cloud_cwp': 0.049100000000000005,
  'manual_cloud_cwp_expr': '49.1 / 1000',
  'manual_cloud_cth': 0.62,
  'manual_cloud_cbh': 0.29,
  'manual_cloud_cot': 8.93,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 8, 9),\n'
                   '                        tmhr_ranges_select=[\n'
                   '                                            [16.029, 16.224], # 100m, cloudy\n'
                   '                                            ],\n'
                   "                        case_tag='cloudy_atm_corr_2',\n"
                   '                        config=config,\n'
                   '                        levels=np.concatenate((np.array([0.0, 0.1, 0.2, 0.29, 0.4, 0.62, 0.8, '
                   '1.0,]),\n'
                   '                                               np.array([1.5, 2.0, 2.5, 3.0, 4.0]), \n'
                   '                                               np.arange(5.0, 10.1, 2.5),\n'
                   '                                               np.array([15, 20, 30., 40., 45.]))),\n'
                   '                        simulation_interval=0.5,\n'
                   '                        clear_sky=False,\n'
                   '                        overwrite_lrt=atm_corr_overwrite_lrt,\n'
                   '                        manual_cloud=True,\n'
                   '                        manual_cloud_cer=8.3,\n'
                   '                        manual_cloud_cwp=49.10/1000,\n'
                   '                        manual_cloud_cth=0.62,\n'
                   '                        manual_cloud_cbh=0.29,\n'
                   '                        manual_cloud_cot=8.93,\n'
                   '                        iter=iter,\n'
                   '                        )'},
 {'id': 'case_069',
  'date': '2024-08-15',
  'case_tag': 'clear_atm_corr',
  'tmhr_ranges_select': [[14.085, 14.396], [14.55, 14.968], [15.078, 15.163]],
  'simulation_interval': 0.5,
  'clear_sky': True,
  'manual_cloud': False,
  'has_custom_levels': False,
  'manual_cloud_cer': 0.0,
  'manual_cloud_cwp': 0.0,
  'manual_cloud_cwp_expr': '0.0',
  'manual_cloud_cth': 0.0,
  'manual_cloud_cbh': 0.0,
  'manual_cloud_cot': 0.0,
  'original_call': 'flt_trk_atm_corr(date=datetime.datetime(2024, 8, 15),\n'
                   '                        tmhr_ranges_select=[\n'
                   '                                            [14.085, 14.396], # 100m, clear\n'
                   '                                            [14.550, 14.968], # 3.5km, clear\n'
                   '                                            [15.078, 15.163], # 1.7km, clear\n'
                   '                                            ],\n'
                   "                        case_tag='clear_atm_corr',\n"
                   '                        config=config,\n'
                   '                        simulation_interval=0.5,\n'
                   '                        clear_sky=True,\n'
                   '                        overwrite_lrt=atm_corr_overwrite_lrt,\n'
                   '                        manual_cloud=False,\n'
                   '                        manual_cloud_cer=0.0,\n'
                   '                        manual_cloud_cwp=0.0,\n'
                   '                        manual_cloud_cth=0.0,\n'
                   '                        manual_cloud_cbh=0.0,\n'
                   '                        manual_cloud_cot=0.0,\n'
                   '                        iter=iter,\n'
                   '                        )'}]

SPIRAL_CASE_CATALOG = [{'id': 'spiral_001',
  'date': '2024-06-06',
  'case_tag': 'clear_sky_spiral_atm_corr',
  'tmhr_ranges_select': [[17.0833, 17.1028],
                         [17.1264, 17.1333],
                         [17.1542, 17.1625],
                         [17.1833, 17.1931],
                         [17.2153, 17.2181],
                         [17.2403, 17.25]],
  'original_call': 'atm_corr_spiral_plot(date=datetime.datetime(2024, 6, 6),\n'
                   '                        tmhr_ranges_select=[[17.0833, 17.1028],\n'
                   '                                            [17.1264, 17.1333],\n'
                   '                                            [17.1542, 17.1625],\n'
                   '                                            [17.1833, 17.1931],\n'
                   '                                            [17.2153, 17.2181],\n'
                   '                                            [17.2403, 17.2500],\n'
                   '                                            ],\n'
                   "                    case_tag='clear_sky_spiral_atm_corr',\n"
                   '                    config=config,\n'
                   '                    )'},
 {'id': 'spiral_002',
  'date': '2024-06-11',
  'case_tag': 'clear_sky_spiral_atm_corr',
  'tmhr_ranges_select': [[14.5667, 14.5694],
                         [14.5986, 14.6097],
                         [14.6375, 14.6486],
                         [14.6778, 14.6903],
                         [14.7208, 14.7403],
                         [14.7653, 14.7875],
                         [14.8125, 14.8278],
                         [14.8542, 14.8736],
                         [14.8986, 14.9389]],
  'original_call': 'atm_corr_spiral_plot(date=datetime.datetime(2024, 6, 11),\n'
                   '                        tmhr_ranges_select=[[14.5667, 14.5694],\n'
                   '                                            [14.5986, 14.6097],\n'
                   '                                            [14.6375, 14.6486], # cloud shadow\n'
                   '                                            [14.6778, 14.6903],\n'
                   '                                            [14.7208, 14.7403],\n'
                   '                                            [14.7653, 14.7875],\n'
                   '                                            [14.8125, 14.8278],\n'
                   '                                            [14.8542, 14.8736],\n'
                   '                                            [14.8986, 14.9389], # more cracks\n'
                   '                                            ],\n'
                   "                    case_tag='clear_sky_spiral_atm_corr',\n"
                   '                    config=config,\n'
                   '                    )'},
 {'id': 'spiral_003',
  'date': '2024-05-31',
  'case_tag': 'clear_sky_spiral_atm_corr',
  'tmhr_ranges_select': [[15.1903, 15.2083],
                         [15.2389, 15.2528],
                         [15.2806, 15.3014],
                         [15.3292, 15.3431],
                         [15.3694, 15.3944],
                         [15.4167, 15.4458],
                         [15.4736, 15.5056],
                         [15.5264, 15.5556],
                         [15.5792, 15.6056],
                         [15.6486, 15.6636],
                         [15.6878, 15.7042]],
  'original_call': 'atm_corr_spiral_plot(date=datetime.datetime(2024, 5, 31),\n'
                   '                        tmhr_ranges_select=[[15.1903, 15.2083],\n'
                   '                                            [15.2389, 15.2528],\n'
                   '                                            [15.2806, 15.3014],\n'
                   '                                            [15.3292, 15.3431],\n'
                   '                                            [15.3694, 15.3944],\n'
                   '                                            [15.4167, 15.4458],\n'
                   '                                            [15.4736, 15.5056],\n'
                   '                                            [15.5264, 15.5556],\n'
                   '                                            [15.5792, 15.6056],\n'
                   '                                            [15.6486, 15.6636],\n'
                   '                                            [15.6878, 15.7042],\n'
                   '                                            ],\n'
                   "                    case_tag='clear_sky_spiral_atm_corr',\n"
                   '                    config=config,\n'
                   '                    )'},
 {'id': 'spiral_004',
  'date': '2024-06-05',
  'case_tag': 'clear_sky_spiral_atm_corr_R0',
  'tmhr_ranges_select': [[13.7889, 13.801],
                         [13.835, 13.8395],
                         [13.878, 13.8885],
                         [13.924, 13.9255],
                         [13.954, 13.9715],
                         [13.998, 14.0153],
                         [14.0417, 14.0475],
                         [14.056, 14.059],
                         [14.0825, 14.0975],
                         [14.1264, 14.1525],
                         [14.1762, 14.1975],
                         [14.2194, 14.242],
                         [14.2605, 14.281]],
  'original_call': 'atm_corr_spiral_plot(date=datetime.datetime(2024, 6, 5),\n'
                   '                    tmhr_ranges_select=[[13.7889, 13.8010],\n'
                   '                                            [13.8350, 13.8395],\n'
                   '                                            [13.8780, 13.8885],\n'
                   '                                            [13.9240, 13.9255],\n'
                   '                                            # [13.9389, 13.9403],\n'
                   '                                            [13.9540, 13.9715],\n'
                   '                                            [13.9980, 14.0153],\n'
                   '                                            # [14.0417, 14.0575],\n'
                   '                                            [14.0417, 14.0475],\n'
                   '                                            [14.0560, 14.0590],\n'
                   '                                            [14.0825, 14.0975],\n'
                   '                                            [14.1264, 14.1525],\n'
                   '                                            [14.1762, 14.1975],\n'
                   '                                            [14.2194, 14.2420],\n'
                   '                                            [14.2605, 14.2810]\n'
                   '                                            ],\n'
                   "                    case_tag='clear_sky_spiral_atm_corr_R0',\n"
                   '                    config=config,\n'
                   '                    )'},
 {'id': 'spiral_005',
  'date': '2024-06-05',
  'case_tag': 'clear_sky_spiral_atm_corr_R1',
  'tmhr_ranges_select': [[13.7889, 13.801],
                         [13.835, 13.8395],
                         [13.878, 13.8885],
                         [13.954, 13.9715],
                         [13.998, 14.0153],
                         [14.0417, 14.0475],
                         [14.056, 14.059],
                         [14.0825, 14.0975],
                         [14.1264, 14.1525],
                         [14.1762, 14.1975],
                         [14.2194, 14.242],
                         [14.2605, 14.281]],
  'original_call': 'atm_corr_spiral_plot(date=datetime.datetime(2024, 6, 5),\n'
                   '                    tmhr_ranges_select=[[13.7889, 13.8010],\n'
                   '                                            [13.8350, 13.8395],\n'
                   '                                            [13.8780, 13.8885],\n'
                   '                                            # [13.9240, 13.9255],\n'
                   '                                            # [13.9389, 13.9403],\n'
                   '                                            [13.9540, 13.9715],\n'
                   '                                            [13.9980, 14.0153],\n'
                   '                                            # [14.0417, 14.0575],\n'
                   '                                            [14.0417, 14.0475],\n'
                   '                                            [14.0560, 14.0590],\n'
                   '                                            [14.0825, 14.0975],\n'
                   '                                            [14.1264, 14.1525],\n'
                   '                                            [14.1762, 14.1975],\n'
                   '                                            [14.2194, 14.2420],\n'
                   '                                            [14.2605, 14.2810]\n'
                   '                                            ],\n'
                   "                    case_tag='clear_sky_spiral_atm_corr_R1',\n"
                   '                    config=config,\n'
                   '                    )'}]


PRE_SURFACE_ALBEDO_CASE_IDS = {
    f'case_{case_number:03d}' for case_number in range(1, 29)
}

# Cases recovered from legacy/ssfr_atm_corr_ori.py before line 2100 are kept
# above in ALL_CASE_CATALOG for reference, but they are not active here.
CASE_CATALOG = [
    case for case in ALL_CASE_CATALOG
    if case['id'] not in PRE_SURFACE_ALBEDO_CASE_IDS
]


def cases_for_date(date_s):
    """Return flight-track catalog entries for a YYYYMMDD or YYYY-MM-DD date string."""
    normalized = date_s if '-' in date_s else f'{date_s[:4]}-{date_s[4:6]}-{date_s[6:8]}'
    return [case for case in CASE_CATALOG if case['date'] == normalized]


def spiral_cases_for_date(date_s):
    """Return legacy spiral-plot catalog entries for a YYYYMMDD or YYYY-MM-DD date string."""
    normalized = date_s if '-' in date_s else f'{date_s[:4]}-{date_s[4:6]}-{date_s[6:8]}'
    return [case for case in SPIRAL_CASE_CATALOG if case['date'] == normalized]


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
    """Run a catalog case that does not require custom levels."""
    case = get_case(case_id)
    if case['has_custom_levels']:
        raise ValueError(f"{case_id} used custom levels in the original script; see case['original_call'].")
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

    def run_final_iteration(iter, final_status):
        if not run_final_sim:
            return
        flt_trk_atm_corr(
            date=datetime.datetime(year, month, day),
            tmhr_ranges_select=case['tmhr_ranges_select'],
            case_tag=case['case_tag'],
            config=config,
            simulation_interval=case['simulation_interval'],
            clear_sky=case['clear_sky'],
            overwrite_lrt=overwrite_lrt,
            manual_cloud=case['manual_cloud'],
            manual_cloud_cer=case['manual_cloud_cer'] or 0.0,
            manual_cloud_cwp=case['manual_cloud_cwp'] or 0.0,
            manual_cloud_cth=case['manual_cloud_cth'] or 0.0,
            manual_cloud_cbh=case['manual_cloud_cbh'] or 0.0,
            manual_cloud_cot=case['manual_cloud_cot'] or 0.0,
            iter=iter,
            final_sim=True,
            final_status=final_status,
        )

    for iter in iterations:
        flt_trk_atm_corr(
            date=datetime.datetime(year, month, day),
            tmhr_ranges_select=case['tmhr_ranges_select'],
            case_tag=case['case_tag'],
            config=config,
            simulation_interval=case['simulation_interval'],
            clear_sky=case['clear_sky'],
            overwrite_lrt=overwrite_lrt,
            manual_cloud=case['manual_cloud'],
            manual_cloud_cer=case['manual_cloud_cer'] or 0.0,
            manual_cloud_cwp=case['manual_cloud_cwp'] or 0.0,
            manual_cloud_cth=case['manual_cloud_cth'] or 0.0,
            manual_cloud_cbh=case['manual_cloud_cbh'] or 0.0,
            manual_cloud_cot=case['manual_cloud_cot'] or 0.0,
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
