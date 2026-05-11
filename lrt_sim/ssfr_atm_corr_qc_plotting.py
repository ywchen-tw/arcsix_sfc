"""Plotting helpers for SSFR atmospheric correction."""

import matplotlib.pyplot as plt
import numpy as np

try:
    from .ssfr_atm_corr_helpers import ssfr_flags
except ImportError:
    from ssfr_atm_corr_helpers import ssfr_flags


def ssfr_time_series_plot(data_hsk, data_ssfr, data_hsr1, tmhr_ranges_select, date_s, case_tag, pitch_roll_thres=3.0):
    t_hsk = np.array(data_hsk["tmhr"])
    t_ssfr = data_ssfr['time'] / 3600.0
    t_hsr1 = data_hsr1['time'] / 3600.0

    pitch_ang = data_hsk['ang_pit']
    roll_ang = data_hsk['ang_rol']

    tmhr_hsk_mask = (t_hsk >= tmhr_ranges_select[0][0]) & (t_hsk <= tmhr_ranges_select[-1][1])

    hsr_wvl = data_hsr1['wvl_dn_tot']
    hsr_ftot = data_hsr1['f_dn_tot']
    hsr_fdif = data_hsr1['f_dn_dif']
    hsr_dif_ratio = hsr_fdif / hsr_ftot
    hsr_550_ind = np.argmin(np.abs(hsr_wvl - 550))
    hsr_530_ind = np.argmin(np.abs(hsr_wvl - 530))
    hsr_570_ind = np.argmin(np.abs(hsr_wvl - 570))
    hsr1_diff_ratio = data_hsr1["f_dn_dif"] / data_hsr1["f_dn_tot"]
    hsr1_diff_ratio_530_570_mean = np.nanmean(hsr1_diff_ratio[:, hsr_530_ind:hsr_570_ind + 1], axis=1)

    pitch_roll_mask = np.sqrt(pitch_ang**2 + roll_ang**2) < pitch_roll_thres
    pitch_ang_valid = pitch_ang.copy()
    roll_ang_valid = roll_ang.copy()
    pitch_ang_valid[~pitch_roll_mask] = np.nan
    roll_ang_valid[~pitch_roll_mask] = np.nan
    pitch_roll_mask = pitch_roll_mask[tmhr_hsk_mask]

    ssfr_zen_wvl = data_ssfr['wvl_dn']
    ssfr_fdn = data_ssfr['f_dn']
    ssfr_nad_wvl = data_ssfr['wvl_up']
    ssfr_fup = data_ssfr['f_up']

    t_ssfr_tmhr_mask = (t_ssfr >= tmhr_ranges_select[0][0]) & (t_ssfr <= tmhr_ranges_select[-1][1])
    t_ssfr_tmhr = t_ssfr[t_ssfr_tmhr_mask]
    ssfr_fdn_tmhr = ssfr_fdn[t_ssfr_tmhr_mask]
    ssfr_fdn_tmhr[~pitch_roll_mask] = np.nan
    ssfr_fup_tmhr = ssfr_fup[t_ssfr_tmhr_mask]
    ssfr_fup_tmhr[~pitch_roll_mask] = np.nan

    t_hsr1_tmhr_mask = (t_hsr1 >= tmhr_ranges_select[0][0]) & (t_hsr1 <= tmhr_ranges_select[-1][1])
    t_hsr1_tmhr = t_hsr1[t_hsr1_tmhr_mask]
    hsr_dif_ratio = hsr_dif_ratio[t_hsr1_tmhr_mask]
    hsr1_diff_ratio_530_570_mean = hsr1_diff_ratio_530_570_mean[t_hsr1_tmhr_mask]

    alp_ang_pit_rol_issue = (data_ssfr['flag'] & ssfr_flags.alp_ang_pit_rol_issue) != 0
    alp_ang_pit_rol_issue_tmhr = alp_ang_pit_rol_issue[t_ssfr_tmhr_mask]
    ssfr_fup_tmhr[alp_ang_pit_rol_issue_tmhr] = np.nan
    ssfr_fdn_tmhr[alp_ang_pit_rol_issue_tmhr] = np.nan

    ssfr_zen_550_ind = np.argmin(np.abs(ssfr_zen_wvl - 550))
    ssfr_nad_550_ind = np.argmin(np.abs(ssfr_nad_wvl - 550))

    time_start, time_end = t_ssfr_tmhr[0], t_ssfr_tmhr[-1]

    fig, (ax10, ax20) = plt.subplots(2, 1, figsize=(16, 12))
    ax11 = ax10.twinx()

    l1 = ax10.plot(t_ssfr, ssfr_fdn[:, ssfr_zen_550_ind], '--', color='k', alpha=0.85, label='SSFR Down 550nm all')
    l2 = ax10.plot(t_ssfr, ssfr_fup[:, ssfr_nad_550_ind], '--', color='k', alpha=0.85, label='SSFR Up 550nm all')

    ax10.plot(t_ssfr_tmhr, ssfr_fdn_tmhr[:, ssfr_zen_550_ind], 'r-', label='SSFR Down 550nm', linewidth=3)
    ax10.plot(t_ssfr_tmhr, ssfr_fup_tmhr[:, ssfr_nad_550_ind], 'b-', label='SSFR Up 550nm', linewidth=3)

    l3 = ax11.plot(t_hsr1_tmhr, hsr_dif_ratio[:, hsr_550_ind], 'm-', label='HSR1 Diff Ratio 550nm')
    ax11.set_ylabel('HSR1 Diff Ratio 550nm', fontsize=14)

    ax20.plot(t_hsk, pitch_ang, 'g-', label='HSK Pitch', linewidth=1.0)
    ax20.plot(t_hsk, roll_ang, 'b-', label='HSK Roll', linewidth=1.0)
    ax20.plot(t_hsk, pitch_ang_valid, 'g-', linewidth=3.0)
    ax20.plot(t_hsk, roll_ang_valid, 'b-', linewidth=3.0)
    for _, (lo, hi) in enumerate(tmhr_ranges_select):
        for ax in [ax10, ax20]:
            ax.fill_betweenx([0, 6], lo, hi, color='gray', alpha=0.3, transform=ax.get_xaxis_transform())
            ax.set_xlabel('Time (UTC)')
            ax.set_xlim(tmhr_ranges_select[0][0] * 0.999, tmhr_ranges_select[-1][1] * 1.001)
    ll = l1 + l2 + l3
    labs = [line.get_label() for line in ll]
    ax10.legend(ll, labs, fontsize=10)
    ax20.legend()
    ax10.set_ylim(0.4, 1.2)
    ax11.set_ylim(0.0, 0.65)
    ax10.set_ylabel('SSFR Flux (W/m$^2$/nm)')

    ax20.set_ylabel('HSK Pitch/Roll (deg)')
    ax20.set_ylim(-5, 5)
    fig.tight_layout()
    fig.savefig(
        'fig/%s/%s_%s_ssfr_pitch_roll_heading_550nm_time_%.2f-%.2f.png'
        % (date_s, date_s, case_tag, time_start, time_end),
        bbox_inches='tight',
        dpi=150,
    )
