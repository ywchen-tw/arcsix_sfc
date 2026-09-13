"""Small helpers for SSFR atmospheric correction."""

import os
import re
import uuid
from contextlib import contextmanager
from enum import IntFlag, auto

import h5py
import numpy as np
import pandas as pd

try:
    from .settings import (
        final_end,
        final_start,
        h2o_1_end,
        h2o_1_start,
        h2o_2_end,
        h2o_2_start,
        h2o_3_end,
        h2o_3_start,
        h2o_4_end,
        h2o_4_start,
        h2o_5_end,
        h2o_5_start,
        h2o_6_end,
        h2o_6_start,
        h2o_7_end,
        h2o_7_start,
        h2o_8_end,
        h2o_8_start,
        o2a_1_end,
        o2a_1_start,
    )
except ImportError:
    from settings import (
        final_end,
        final_start,
        h2o_1_end,
        h2o_1_start,
        h2o_2_end,
        h2o_2_start,
        h2o_3_end,
        h2o_3_start,
        h2o_4_end,
        h2o_4_start,
        h2o_5_end,
        h2o_5_start,
        h2o_6_end,
        h2o_6_start,
        h2o_7_end,
        h2o_7_start,
        h2o_8_end,
        h2o_8_start,
        o2a_1_end,
        o2a_1_start,
    )


# SSFR R1 data-quality flag. Bit index = declaration order, mirroring `ssfr_flags`
# in ssfr/projects/2024-arcsix/_arcsix_archive.py (layout of 2026-09-11). The R1
# file documents its own layout in the `flag` long_name; call
# `verify_ssfr_flag_layout` before decoding so a re-ordered product can never be
# read with a stale enum.
class ssfr_flags(IntFlag):
    tec_temp_controller_issue = auto()        # bit 0: thermoelectric cooler (TEC) temperature controller issue
    hsk_pitch_roll_exceed_threshold = auto()  # bit 1: aircraft (HSK INS) pitch/roll exceeded threshold (nadir collector)
    camera_icing = auto()                     # bit 2: camera icing at measurement time
    camera_icing_pre = auto()                 # bit 3: camera icing within 30 minutes prior to measurement time
    zen_toa_over_threshold = auto()           # bit 4: zenith flux exceeded the TOA irradiance threshold
    alp_hsk_ang_issue = auto()                # bit 5: leveling-platform pitch/roll rate-of-change issue
    alp_pitch_roll_exceed_threshold = auto()  # bit 6: zenith collector tilt after platform compensation exceeded threshold


# Bits that preprocess masks on by default (zenith, nadir and TOA together).
# Bits 2, 3, 5 and 6 are recorded in the leg pickle but not masked.
ssfr_default_mask_flags = (
    ssfr_flags.tec_temp_controller_issue,
    ssfr_flags.hsk_pitch_roll_exceed_threshold,
    ssfr_flags.zen_toa_over_threshold,
)

# Keyword that must appear in the R1 long_name description of each bit.
_SSFR_FLAG_KEYWORDS = {
    ssfr_flags.tec_temp_controller_issue: 'thermoelectric cooler',
    ssfr_flags.hsk_pitch_roll_exceed_threshold: 'aircraft pitch/roll',
    ssfr_flags.camera_icing: 'camera icing at measurement time',
    ssfr_flags.camera_icing_pre: 'camera icing within',
    ssfr_flags.zen_toa_over_threshold: 'exceeding the toa',
    ssfr_flags.alp_hsk_ang_issue: 'rate-of-change',
    ssfr_flags.alp_pitch_roll_exceed_threshold: 'light-collector tilt',
}


def ssfr_flag_bit(flag_member):
    """Return the 0-based bit index of an ``ssfr_flags`` member."""
    return int(flag_member.value).bit_length() - 1


def resolve_ssfr_mask_flags(mask_flags=None):
    """Normalise ``mask_flags`` (members, names, or None) to a tuple of members."""
    if mask_flags is None:
        return ssfr_default_mask_flags
    resolved = []
    for item in mask_flags:
        if isinstance(item, ssfr_flags):
            resolved.append(item)
        elif isinstance(item, str):
            resolved.append(ssfr_flags[item])
        else:
            raise TypeError(f'mask_flags entries must be ssfr_flags members or names, got {item!r}')
    return tuple(resolved)


def verify_ssfr_flag_layout(fname_ssfr):
    """Check that the ``flag`` long_name in an SSFR R1 file matches ``ssfr_flags``.

    The archive script composes the long_name as ``bit N: description`` entries in
    enum order, so a re-ordered or extended product is caught here instead of
    silently decoding the wrong bits. Returns the long_name on success.
    """
    with h5py.File(fname_ssfr, 'r') as f:
        long_name = f['flag'].attrs.get('long_name', '')
    if isinstance(long_name, bytes):
        long_name = long_name.decode()
    long_name = str(long_name)

    described = {
        int(m.group(1)): m.group(2).strip()
        for m in re.finditer(r'bit\s+(\d+):\s*([^;]+)', long_name)
    }

    problems = []
    for member, keyword in _SSFR_FLAG_KEYWORDS.items():
        bit = ssfr_flag_bit(member)
        text = described.get(bit)
        if text is None:
            problems.append(f'bit {bit} ({member.name}) is not documented in the file')
        elif keyword not in text.lower():
            problems.append(f'bit {bit} ({member.name}) reads "{text}" in the file')
    extra = sorted(set(described) - {ssfr_flag_bit(m) for m in _SSFR_FLAG_KEYWORDS})
    if extra:
        problems.append(f'file documents bits {extra} unknown to ssfr_flags')
    if problems:
        raise RuntimeError(
            f'SSFR flag layout mismatch in {fname_ssfr}: ' + '; '.join(problems)
            + f'. File long_name: "{long_name}"'
        )
    return long_name


@contextmanager
def atomic_write(filename, mode='w', **open_kwargs):
    """Write to a temp file in the same directory, then atomically replace `filename`.

    Guards against torn/interleaved output when more than one process targets the
    same path (e.g. a parallel run overlapping a serial one): each writer commits a
    complete file via os.replace, so a race can only ever clobber the target with
    another *complete* file, never corrupt it mid-write.
    """
    directory = os.path.dirname(filename) or '.'
    os.makedirs(directory, exist_ok=True)
    tmp = os.path.join(
        directory,
        f'.{os.path.basename(filename)}.{os.getpid()}.{uuid.uuid4().hex}.tmp',
    )
    try:
        with open(tmp, mode, **open_kwargs) as f:
            yield f
        os.replace(tmp, filename)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def write_2col_file(filename, wvl, val, header):
    """Write wavelength/value arrays as a two-column text file."""
    with atomic_write(filename) as f:
        f.write(header)
        for i in range(len(val)):
            f.write(f'{wvl[i]:11.3f} {val[i]:12.3e}\n')


def find_h2o_6_end(wvl, alb, default_end=h2o_6_end):
    """Extend the H2O-6 mask until albedo recovers to its pre-band value."""
    wvl = np.asarray(wvl, dtype=float)
    alb = np.asarray(alb, dtype=float)
    finite = np.isfinite(wvl) & np.isfinite(alb)

    before_indices = np.flatnonzero(finite & (wvl < h2o_6_start))
    after_indices = np.flatnonzero(finite & (wvl > default_end))
    if before_indices.size == 0 or after_indices.size == 0:
        return default_end

    reference_albedo = alb[before_indices[-1]]
    first_after_index = after_indices[0]
    if alb[first_after_index] <= reference_albedo:
        return default_end

    recovered_indices = after_indices[alb[after_indices] <= reference_albedo]
    if recovered_indices.size == 0:
        return default_end
    return float(wvl[recovered_indices[0]])


def gas_abs_masking(
    wvl,
    alb,
    alt,
    altitude_dependent=False,
    h2o_6_end_override=None,
):
    """Mask all gas bands by default, with optional reduced low-altitude masking."""
    effective_mask_ = np.ones_like(alb)
    alb_mask = alb.copy()
    selected_h2o_6_end = h2o_6_end if h2o_6_end_override is None else h2o_6_end_override
    full_mask = (
        ((wvl >= o2a_1_start) & (wvl <= o2a_1_end))
        | ((wvl >= h2o_1_start) & (wvl <= h2o_1_end))
        | ((wvl >= h2o_2_start) & (wvl <= h2o_2_end))
        | ((wvl >= h2o_3_start) & (wvl <= h2o_3_end))
        | ((wvl >= h2o_4_start) & (wvl <= h2o_4_end))
        | ((wvl >= h2o_5_start) & (wvl <= h2o_5_end))
        | ((wvl >= h2o_6_start) & (wvl <= selected_h2o_6_end))
        | ((wvl >= h2o_7_start) & (wvl <= h2o_7_end))
        | ((wvl >= h2o_8_start) & (wvl <= h2o_8_end))
        | ((wvl >= final_start) & (wvl <= final_end))
    )

    # Future option: retain O2 masking but omit short-path H2O bands below 0.5 km.
    reduced_low_altitude_mask = (
        ((wvl >= o2a_1_start) & (wvl <= o2a_1_end))
        | ((wvl >= h2o_3_start) & (wvl <= h2o_3_end))
        | ((wvl >= h2o_4_start) & (wvl <= h2o_4_end))
        | ((wvl >= h2o_5_start) & (wvl <= h2o_5_end))
        | ((wvl >= h2o_6_start) & (wvl <= selected_h2o_6_end))
        | ((wvl >= h2o_7_start) & (wvl <= h2o_7_end))
        | ((wvl >= final_start) & (wvl <= final_end))
    )
    mask = reduced_low_altitude_mask if altitude_dependent and alt <= 0.5 else full_mask

    alb_mask[mask] = np.nan
    effective_mask_[mask] = np.nan

    if np.sum(~np.isnan(effective_mask_)) != np.isfinite(alb_mask).sum():
        fit_wvl_mask = np.logical_and(~np.isnan(effective_mask_), np.isnan(alb_mask))

        s = pd.Series(alb_mask[effective_mask_ == 1])
        s_mask = np.isnan(alb_mask[effective_mask_ == 1])
        s_ffill = s.ffill(limit=2).bfill(limit=2)
        while np.any(np.isnan(s_ffill)):
            s_ffill = s_ffill.ffill(limit=2).bfill(limit=2)

        alb_mask[fit_wvl_mask] = np.array(s_ffill)[s_mask]

    return alb_mask
