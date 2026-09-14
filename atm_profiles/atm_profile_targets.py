"""Targets for standalone atmospheric-profile (atm + ch4) generation.

Each entry drives one ``prepare_atmospheric_profile()`` call. The SIGMA site
label is informational only -- it does not affect the generated profile, which
depends on date, time (MODIS overpass), the aircraft lat/lon box, and the
MARLI/dropsonde/climatology inputs.

This module is data only, mirroring ``lrt_sim/ssfr_atm_corr/case_catalog.py``.
The processing logic lives in ``make_atm_profiles.py``.
"""

import math

# +/- minutes around each observation time used to build the (start, end) window.
HALF_WINDOW_MIN = 5.0

# Case tag written into the output filenames. Times keep entries distinct, so a
# single shared tag is fine; SIGMA A/B is preserved only in the 'site' field.
DEFAULT_CASE_TAG = 'sigma_atm'


ATM_PROFILE_TARGETS = [
    {'id': 'atm_001', 'date': '2024-07-25', 'time': '11:42', 'site': 'SIGMA-B'},
    {'id': 'atm_002', 'date': '2024-07-29', 'time': '11:38', 'site': 'SIGMA-B'},
    {'id': 'atm_003', 'date': '2024-07-29', 'time': '18:24', 'site': 'SIGMA-A'},
    {'id': 'atm_004', 'date': '2024-07-29', 'time': '18:26', 'site': 'SIGMA-B'},
    {'id': 'atm_005', 'date': '2024-07-30', 'time': '11:24', 'site': 'SIGMA-B'},
    {'id': 'atm_006', 'date': '2024-07-30', 'time': '11:29', 'site': 'SIGMA-A'},
    {'id': 'atm_007', 'date': '2024-07-30', 'time': '17:19', 'site': 'SIGMA-A'},
    {'id': 'atm_008', 'date': '2024-08-02', 'time': '11:24', 'site': 'SIGMA-B'},
    {'id': 'atm_009', 'date': '2024-08-02', 'time': '11:32', 'site': 'SIGMA-A'},
    {'id': 'atm_010', 'date': '2024-08-08', 'time': '11:19', 'site': 'SIGMA-B'},
    {'id': 'atm_011', 'date': '2024-08-08', 'time': '11:26', 'site': 'SIGMA-A'},
    {'id': 'atm_012', 'date': '2024-08-15', 'time': '11:27', 'site': 'SIGMA-A'},
]


def get_target(target_id):
    """Return the target dict with the given id (raises KeyError if absent)."""
    for target in ATM_PROFILE_TARGETS:
        if target['id'] == target_id:
            return target
    raise KeyError(f"No atm-profile target with id {target_id!r}")


def time_to_hours(hhmm):
    """Convert an 'HH:MM' string to decimal hours."""
    hours, minutes = hhmm.split(':')
    return int(hours) + int(minutes) / 60.0


def target_window(target, half_window_min=HALF_WINDOW_MIN):
    """Return (time_start, time_end) in decimal hours around the target time."""
    mid = time_to_hours(target['time'])
    half = half_window_min / 60.0
    return mid - half, mid + half
