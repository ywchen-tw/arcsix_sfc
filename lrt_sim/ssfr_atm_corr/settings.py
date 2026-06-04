"""Configuration constants for SSFR atmospheric correction."""

import datetime
import platform

_mission_ = 'arcsix'
_platform_ = 'p3b'

_hsk_ = 'hsk'
_alp_ = 'alp'
_spns_ = 'spns-a'
_ssfr1_ = 'ssfr-a'
_ssfr2_ = 'ssfr-b'
_cam_ = 'nac'

_fdir_main_ = 'data/flt-vid'
_fdir_sat_img_ = 'data/%s/sat-img' % _mission_
_fdir_sat_data_ = 'data/%s/sat' % _mission_
_fdir_cam_img_ = 'data/%s/2024-Spring/p3' % _mission_
_wavelength_ = 555.0

_fdir_sat_img_vn_ = 'data/%s/sat-img-vn' % _mission_

_preferred_region_ = 'ca_archipelago'
_aspect_ = 'equal'

if platform.system() == 'Darwin':
    _fdir_data_ = '../data/processed'
    _fdir_general_ = '../data'
    _fdir_tmp_ = './tmp'
elif platform.system() == 'Linux':
    _fdir_data_ = "/pl/active/vikas-arcsix/yuch8913/arcsix/data/processed"
    _fdir_general_ = "/pl/active/vikas-arcsix/yuch8913/arcsix/data"
    _fdir_tmp_ = "/pl/active/vikas-arcsix/yuch8913/arcsix/tmp"
else:
    _fdir_data_ = '../data/processed'
    _fdir_general_ = '../data'
    _fdir_tmp_ = './tmp'

_fdir_tmp_graph_ = 'tmp-graph_flt-vid'
_title_extra_ = 'ARCSIX RF#1'

_tmhr_range_ = {
    '20240517': [19.20, 23.00],
    '20240521': [14.80, 17.50],
}

_dates1_ = [
    datetime.datetime(2024, 5, 28),
    datetime.datetime(2024, 5, 30),
    datetime.datetime(2024, 5, 31),
    datetime.datetime(2024, 6, 3),
    datetime.datetime(2024, 6, 5),
    datetime.datetime(2024, 6, 6),
    datetime.datetime(2024, 6, 7),
    datetime.datetime(2024, 6, 10),
    datetime.datetime(2024, 6, 11),
    datetime.datetime(2024, 6, 13),
]

_dates2_ = [
    datetime.datetime(2024, 7, 25),
    datetime.datetime(2024, 7, 29),
    datetime.datetime(2024, 7, 30),
    datetime.datetime(2024, 8, 1),
    datetime.datetime(2024, 8, 2),
    datetime.datetime(2024, 8, 7),
    datetime.datetime(2024, 8, 8),
    datetime.datetime(2024, 8, 9),
    datetime.datetime(2024, 8, 15),
]

_dates_ = _dates1_

o2a_1_start, o2a_1_end = 748, 780
h2o_1_start, h2o_1_end = 650, 706
h2o_2_start, h2o_2_end = 705, 760
h2o_3_start, h2o_3_end = 884, 996
h2o_4_start, h2o_4_end = 1084, 1175
h2o_5_start, h2o_5_end = 1230, 1286
h2o_6_start, h2o_6_end = 1290, 1509
h2o_7_start, h2o_7_end = 1748, 2050
h2o_8_start, h2o_8_end = 801, 843
final_start, final_end = 2110, 2200

gas_bands = [
    (o2a_1_start, o2a_1_end),
    (h2o_1_start, h2o_1_end),
    (h2o_2_start, h2o_2_end),
    (h2o_3_start, h2o_3_end),
    (h2o_4_start, h2o_4_end),
    (h2o_5_start, h2o_5_end),
    (h2o_6_start, h2o_6_end),
    (h2o_7_start, h2o_7_end),
    (h2o_8_start, h2o_8_end),
    (final_start, final_end),
]

# Future experiment switch. Keep False to mask every configured gas band at all altitudes.
ALTITUDE_DEPENDENT_GAS_MASKING = False

__all__ = [
    name
    for name in globals()
    if not name.startswith('__') and name not in ('datetime', 'platform')
]
