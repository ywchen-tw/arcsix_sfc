import os
import sys
import glob
import copy
import pickle
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt

# mpl.use('Agg')


def gas_abs_masking(wvl, alb, alt, h2o_6_end=1509, interp_nan=True):
    o2a_1_start, o2a_1_end = 748, 780
    # h2o_1_start, h2o_1_end = 672, 706
    # h2o_2_start, h2o_2_end = 705, 746
    h2o_1_start, h2o_1_end = 650 , 706
    h2o_2_start, h2o_2_end = 705, 760
    h2o_3_start, h2o_3_end = 884, 996
    h2o_4_start, h2o_4_end = 1084, 1175
    h2o_5_start, h2o_5_end = 1230, 1286
    h2o_6_start, h2o_6_end = 1290, h2o_6_end
    h2o_7_start, h2o_7_end = 1748, 2050
    h2o_8_start, h2o_8_end = 801, 843
    final_start, final_end = 2110, 2200
    
    effective_mask_ = np.ones_like(alb)
    alb_mask = alb.copy()
    if 1:#alt > 0.5:
        alb_mask[
                ((wvl>=o2a_1_start) & (wvl<=o2a_1_end)) | 
                ((wvl>=h2o_1_start) & (wvl<=h2o_1_end)) | 
                ((wvl>=h2o_2_start) & (wvl<=h2o_2_end)) | 
                ((wvl>=h2o_3_start) & (wvl<=h2o_3_end)) | 
                ((wvl>=h2o_4_start) & (wvl<=h2o_4_end)) | 
                ((wvl>=h2o_5_start) & (wvl<=h2o_5_end)) | 
                ((wvl>=h2o_6_start) & (wvl<=h2o_6_end)) | 
                ((wvl>=h2o_7_start) & (wvl<=h2o_7_end)) |
                ((wvl>=h2o_8_start) & (wvl<=h2o_8_end)) |
                ((wvl>=final_start) & (wvl<=final_end))
                ] = np.nan
        effective_mask_[
                ((wvl>=o2a_1_start) & (wvl<=o2a_1_end)) | 
                ((wvl>=h2o_1_start) & (wvl<=h2o_1_end)) | 
                ((wvl>=h2o_2_start) & (wvl<=h2o_2_end)) | 
                ((wvl>=h2o_3_start) & (wvl<=h2o_3_end)) | 
                ((wvl>=h2o_4_start) & (wvl<=h2o_4_end)) | 
                ((wvl>=h2o_5_start) & (wvl<=h2o_5_end)) | 
                ((wvl>=h2o_6_start) & (wvl<=h2o_6_end)) | 
                ((wvl>=h2o_7_start) & (wvl<=h2o_7_end)) |
                ((wvl>=h2o_8_start) & (wvl<=h2o_8_end)) |
                ((wvl>=final_start) & (wvl<=final_end))
                ] = np.nan
    elif alt <= 0.5 and alt > 0.2:
        alb_mask[
                # ((wvl>=o2a_1_start) & (wvl<=o2a_1_end)) | 
                # ((wvl>=h2o_1_start) & (wvl<=h2o_1_end)) | 
                # ((wvl>=h2o_2_start) & (wvl<=h2o_2_end)) | 
                ((wvl>=h2o_3_start) & (wvl<=h2o_3_end)) | 
                ((wvl>=h2o_4_start) & (wvl<=h2o_4_end)) | 
                ((wvl>=h2o_5_start) & (wvl<=h2o_5_end)) | 
                ((wvl>=h2o_6_start) & (wvl<=h2o_6_end)) | 
                ((wvl>=h2o_7_start) & (wvl<=h2o_7_end)) |
                # ((wvl>=h2o_8_start) & (wvl<=h2o_8_end)) |
                ((wvl>=final_start) & (wvl<=final_end))
                ] = np.nan
        effective_mask_[
                # ((wvl>=o2a_1_start) & (wvl<=o2a_1_end)) | 
                # ((wvl>=h2o_1_start) & (wvl<=h2o_1_end)) | 
                # ((wvl>=h2o_2_start) & (wvl<=h2o_2_end)) | 
                ((wvl>=h2o_3_start) & (wvl<=h2o_3_end)) | 
                ((wvl>=h2o_4_start) & (wvl<=h2o_4_end)) | 
                ((wvl>=h2o_5_start) & (wvl<=h2o_5_end)) | 
                ((wvl>=h2o_6_start) & (wvl<=h2o_6_end)) | 
                ((wvl>=h2o_7_start) & (wvl<=h2o_7_end)) |
                # ((wvl>=h2o_8_start) & (wvl<=h2o_8_end)) |
                ((wvl>=final_start) & (wvl<=final_end))
                ] = np.nan
    else: 
        # Not mask O2 band and water abs band at VIS and NIR if altitude is low
        alb_mask[
                # ((wvl>=o2a_1_start) & (wvl<=o2a_1_end)) | 
                # ((wvl>=h2o_1_start) & (wvl<=h2o_1_end)) | 
                # ((wvl>=h2o_2_start) & (wvl<=h2o_2_end)) | 
                ((wvl>=h2o_3_start) & (wvl<=h2o_3_end)) | 
                ((wvl>=h2o_4_start) & (wvl<=h2o_4_end)) | 
                ((wvl>=h2o_5_start) & (wvl<=h2o_5_end)) | 
                ((wvl>=h2o_6_start) & (wvl<=h2o_6_end)) | 
                ((wvl>=h2o_7_start) & (wvl<=h2o_7_end)) |
                # ((wvl>=h2o_8_start) & (wvl<=h2o_8_end)) |
                ((wvl>=final_start) & (wvl<=final_end))
                ] = np.nan
        effective_mask_[
                # ((wvl>=o2a_1_start) & (wvl<=o2a_1_end)) | 
                # ((wvl>=h2o_1_start) & (wvl<=h2o_1_end)) | 
                # ((wvl>=h2o_2_start) & (wvl<=h2o_2_end)) | 
                ((wvl>=h2o_3_start) & (wvl<=h2o_3_end)) | 
                ((wvl>=h2o_4_start) & (wvl<=h2o_4_end)) | 
                ((wvl>=h2o_5_start) & (wvl<=h2o_5_end)) | 
                ((wvl>=h2o_6_start) & (wvl<=h2o_6_end)) | 
                ((wvl>=h2o_7_start) & (wvl<=h2o_7_end)) |
                # ((wvl>=h2o_8_start) & (wvl<=h2o_8_end)) |
                ((wvl>=final_start) & (wvl<=final_end))
                ] = np.nan
    
    before_interp = alb_mask.copy()
    # interpolation if nan in effective_mask_ range
    if interp_nan and np.sum(~np.isnan(effective_mask_)) != np.isfinite(alb_mask).sum():
        eff_wvl_real_mask = np.logical_and(~np.isnan(effective_mask_), np.isfinite(alb_mask))
        fit_wvl_mask = np.logical_and(~np.isnan(effective_mask_), np.isnan(alb_mask))
        # effective_mask_func = interp1d(wvl[eff_wvl_real_mask], effective_mask_[eff_wvl_real_mask], bounds_error=False, fill_value=np.nan)
        
        
        # alb_mask[fit_wvl_mask] = effective_mask_func(wvl[fit_wvl_mask])
        
        s = pd.Series(alb_mask[effective_mask_==1])
        s_mask = np.isnan(alb_mask[effective_mask_==1])
        # Fills NaN with the value immediately preceding it
        s_ffill = s.fillna(method='ffill', limit=2)
        s_ffill = s_ffill.fillna(method='bfill', limit=2)
        while np.any(np.isnan(s_ffill)):
            s_ffill = s_ffill.fillna(method='ffill', limit=2)
            s_ffill = s_ffill.fillna(method='bfill', limit=2)
        
        
        alb_mask[fit_wvl_mask] = np.array(s_ffill)[s_mask]
        
    # plt.close('all')
    # plt.plot(wvl, before_interp, '-o', label='Before fill')
    # plt.plot(wvl, alb_mask, '-x', label='After fill')
    # plt.xlabel('Wavelength (nm)')
    # plt.ylabel('Albedo')
    # plt.legend()
    # plt.show()
     
    
    return alb_mask
   



def find_best_fit(model_library, obs_wvl, obs_albedo):
    """
    Finds the best-fit model spectrum from a library by minimizing RMSE
    at *only* the provided obs_wvl points.
    """
    
    best_fit_params = None
    best_fit_spectrum = None
    min_rmse = np.inf
    
    obs_wvl_nanmask = np.isfinite(obs_albedo)
    
    # Loop through every pre-run model spectrum in your library
    for key in model_library.keys():
        
        model_run = model_library[key]
        model_wvl = model_run['wvl']      # Full wvl (0.2-5.0 um)
        model_wvl *= 1000  # Convert to nm
        model_albedo = model_run['albedo']  # Full albedo spectrum
        
        # -----------------------------------------------------------------
        # THIS IS THE KEY STEP:
        # It takes the full model (model_wvl, model_albedo) and
        # interpolates it, pulling out *only* the values at the
        # exact points you have in obs_wvl.
        # -----------------------------------------------------------------
        
        interpolated_model_albedo = np.interp(obs_wvl, model_wvl, model_albedo)
        
        # The gaps (0.9-1.1, 1.3-1.5) are automatically
        # and correctly ignored in the next step.
        
        # 3. Calculate RMSE *only* on the valid, non-gap data
        rmse = np.sqrt(np.mean((interpolated_model_albedo[obs_wvl_nanmask] - obs_albedo[obs_wvl_nanmask])**2))
        
        # 4. Check if this is the best fit so far
        if rmse < min_rmse:
            min_rmse = rmse
            best_fit_params = key
            # best_fit_spectrum = model_run # Store the whole run
            best_fit_spectrum = interpolated_model_albedo # Store the interpolation
        
    # plt.close('all')
    # plt.plot(obs_wvl, best_fit_spectrum, '-', color='r', label='Best Fit Model')
    # plt.plot(obs_wvl, obs_albedo, '-', color='k', label='Observed')
    # plt.xlabel('Wavelength (nm)')
    # plt.ylabel('Albedo')
    # plt.legend()
    # plt.title(f'Best Fit Model: {best_fit_params}, RMSE: {min_rmse:.4f}')
    # plt.show()
    
        
    return best_fit_params, best_fit_spectrum, min_rmse

def snowice_alb_fitting(alb_wvl, alb_corr, alt, clear_sky=False, h2o_6_end=1509):
    # snicar_albedo_list = []
    if clear_sky:
        snicar_filename = 'snicar_model_results_direct.pkl'
    else:
        snicar_filename = 'snicar_model_results_diffuse.pkl'
    with open(snicar_filename, 'rb') as f:
        snicar_data = pickle.load(f)
    #     wvl = list(snicar_data.values())[0]['wvl']
    #     for key in snicar_data:
    #         snicar_albedo_list.append((key, snicar_data[key]['albedo']))
    # snicar_albedo_arr = np.array(snicar_albedo_list)  
          
    alb_corr_mask = alb_corr.copy()
    alb_corr_mask = gas_abs_masking(alb_wvl, alb_corr_mask, alt=alt, h2o_6_end=h2o_6_end)
    best_fit_key, best_fit_spectrum, min_rmse = find_best_fit(
        model_library=snicar_data,
        obs_wvl=alb_wvl,
        obs_albedo=alb_corr_mask
    )
    
    alb_corr_best_fit = np.copy(alb_corr_mask)
    mask_bands = np.isnan(alb_corr_mask)
    alb_corr_best_fit[mask_bands] = best_fit_spectrum[mask_bands]
    
    # plt.close('all')
    # plt.plot(alb_wvl, alb_corr, '-', color='k', label='Corrected Albedo')
    # plt.plot(alb_wvl, alb_corr_mask, '-', color='g', label='Masked Corrected Albedo')
    # plt.plot(alb_wvl, best_fit_spectrum, '-', color='r', label='Best Fitted Albedo')
    # plt.fill_between(alb_wvl, -0.05, 1.05, where=np.isnan(alb_corr_mask), color='gray', alpha=0.2, label='Mask Gas absorption bands')

    # plt.xlabel('Wavelength (nm)')
    # plt.ylabel('Albedo')
    # plt.legend()
    # plt.title(f'SNICAR Best Fit Model: {best_fit_key}, RMSE: {min_rmse:.4f}')
    # plt.show()
    
    
    
    # alb_wvl_sep_1nd_s, alb_wvl_sep_1nd_e = 370, 795
    alb_wvl_sep_1nd_s, alb_wvl_sep_1nd_e = 370, 800
    alb_wvl_sep_2nd_s, alb_wvl_sep_2nd_e = 795, 850
    alb_wvl_sep_3rd_s, alb_wvl_sep_3rd_e = 850, 1050
    alb_wvl_sep_4th_s, alb_wvl_sep_4th_e = 1050, 1210
    alb_wvl_sep_5th_s, alb_wvl_sep_5th_e = 1185, 1700
    alb_wvl_sep_6th_s, alb_wvl_sep_6th_e = 1520, 2100
    if h2o_6_end > 1520:
        alb_wvl_sep_6th_s = h2o_6_end+5
    
    band_1_fit = (alb_wvl >= alb_wvl_sep_1nd_s) & (alb_wvl < alb_wvl_sep_1nd_e)
    band_2_fit = (alb_wvl >= alb_wvl_sep_2nd_s) & (alb_wvl < alb_wvl_sep_2nd_e)
    band_3_fit = (alb_wvl >= alb_wvl_sep_3rd_s) & (alb_wvl < alb_wvl_sep_3rd_e)
    band_4_fit = (alb_wvl >= alb_wvl_sep_4th_s) & (alb_wvl < alb_wvl_sep_4th_e)
    band_5_fit = (alb_wvl >= alb_wvl_sep_5th_s) & (alb_wvl < alb_wvl_sep_5th_e)
    band_6_fit = (alb_wvl >= alb_wvl_sep_6th_s) & (alb_wvl <= alb_wvl_sep_6th_e)
    
    alb_corr_fit = copy.deepcopy(alb_corr_mask)
    
    for bands_fit in [
                      band_1_fit, 
                      ]:
        
        # if np.isnan(alb_corr_mask[bands_fit]).any():
            # best_fit_key, best_fit_spectrum, min_rmse = find_best_fit(
            #     model_library=snicar_data,
            #     obs_wvl=alb_wvl[bands_fit],
            #     obs_albedo=alb_corr_mask[bands_fit]
            #     )
        bandfit_nan = np.isnan(alb_corr_mask[bands_fit])
        if bandfit_nan.sum() == 0:
            continue
        bandfit_nan_ind = np.where(bandfit_nan)[0]
        if bandfit_nan_ind[-1] == len(bandfit_nan)-1:
            bandfit_nan_ind = bandfit_nan_ind[:-1]
        left_mean_ind_num = 5 
        if bandfit_nan_ind[0] < left_mean_ind_num:
            left_mean_ind_num = bandfit_nan_ind[0]
        xl_origin = alb_corr_fit[bands_fit][bandfit_nan_ind[0]-left_mean_ind_num:bandfit_nan_ind[0]-1].mean()
        right_mean_ind_num = 5
        if (len(bandfit_nan) - bandfit_nan_ind[-1] -1) < right_mean_ind_num:
            right_mean_ind_num = len(bandfit_nan) - bandfit_nan_ind[-1] -1
            
        # print("bandfit_nan_ind[-1]:", bandfit_nan_ind[-1])
        # print("len(bandfit_nan):", len(bandfit_nan))
        # print("xr_origin start end:", bandfit_nan_ind[-1]+1, bandfit_nan_ind[-1]+right_mean_ind_num)
        xr_origin = alb_corr_fit[bands_fit][bandfit_nan_ind[-1]+1:bandfit_nan_ind[-1]+right_mean_ind_num].mean()

        wvl550nm_ind = np.argmin(np.abs(alb_wvl[bands_fit][~bandfit_nan]-550))
        fit_2nd = np.poly1d(np.polyfit(alb_wvl[bands_fit][~bandfit_nan][wvl550nm_ind:],
                                        alb_corr_mask[bands_fit][~bandfit_nan][wvl550nm_ind:], 2))
        replace_array = fit_2nd(alb_wvl[bands_fit][bandfit_nan])
        alb_corr_fit_replace = copy.deepcopy(alb_corr_fit[bands_fit])
        alb_corr_fit_replace[bandfit_nan] = copy.deepcopy(replace_array)
        alb_corr_fit[bands_fit] = copy.deepcopy(alb_corr_fit_replace)
        
        # plt.close('all')
        # fig, ax = plt.subplots(figsize=(9, 5))
        # ax.plot(alb_wvl[bands_fit], alb_corr_mask[bands_fit], 'o', color='k', label='Corrected Albedo')
        # ax.plot(alb_wvl[bands_fit], alb_corr_fit_replace, '--', color='b', label='Replace')
        # ax.plot(alb_wvl[bands_fit][bandfit_nan], replace_array, 'x', color='m', label='Fitted Points')
        # ax.plot(alb_wvl[bands_fit], alb_corr_fit[bands_fit], '-', color='r', label='Fitted Albedo')
        # ax.set_xlabel('Wavelength (nm)')
        # ax.set_ylabel('Albedo')
        # ax.legend()
        # plt.show()
        
    
    
    for bands_fit in [
                    #   band_1_fit, 
                      band_2_fit, band_3_fit, band_4_fit, band_5_fit, band_6_fit]:
        
        # if np.isnan(alb_corr_mask[bands_fit]).any():
            # best_fit_key, best_fit_spectrum, min_rmse = find_best_fit(
            #     model_library=snicar_data,
            #     obs_wvl=alb_wvl[bands_fit],
            #     obs_albedo=alb_corr_mask[bands_fit]
            #     )
        bandfit_nan = np.isnan(alb_corr_mask[bands_fit])
        if bandfit_nan.sum() == 0:
            continue
        bandfit_nan_ind = np.where(bandfit_nan)[0]
        if bandfit_nan_ind[-1] == len(bandfit_nan)-1:
            bandfit_nan_ind = bandfit_nan_ind[:-1]
        left_mean_ind_num = 5
        if bandfit_nan_ind[0] < left_mean_ind_num:
            left_mean_ind_num = bandfit_nan_ind[0]
        xl_origin = alb_corr_fit[bands_fit][bandfit_nan_ind[0]-left_mean_ind_num:bandfit_nan_ind[0]-1].mean()
        right_mean_ind_num = 5
        if (len(bandfit_nan) - bandfit_nan_ind[-1] -1) < right_mean_ind_num:
            right_mean_ind_num = len(bandfit_nan) - bandfit_nan_ind[-1] -1
            
        # print("bandfit_nan_ind[-1]:", bandfit_nan_ind[-1])
        # print("len(bandfit_nan):", len(bandfit_nan))
        # print("xr_origin start end:", bandfit_nan_ind[-1]+1, bandfit_nan_ind[-1]+right_mean_ind_num)
        xr_origin = alb_corr_fit[bands_fit][bandfit_nan_ind[-1]+1:bandfit_nan_ind[-1]+right_mean_ind_num].mean()
        xl_fit, xr_fit = best_fit_spectrum[bands_fit][bandfit_nan_ind[0]-1], best_fit_spectrum[bands_fit][bandfit_nan_ind[-1]+1]
        xfit_base = np.min([xl_fit, xr_fit])
        # print("xl_origin, xr_origin:", xl_origin, xr_origin)
        if np.isfinite(xl_origin) and np.isfinite(xr_origin):
            base = np.min([xl_origin, xr_origin])
            scale = (xr_origin - xl_origin) / (xr_fit - xl_fit)
            scale = np.abs(scale)
            replace_array = base + (best_fit_spectrum[bands_fit][bandfit_nan] - xfit_base) * scale
            # print("rescale (1) replace_array shape:", replace_array.shape)
        elif np.isfinite(xl_origin) and not np.isfinite(xr_origin):
            # only have value on xl_origin
            # use valid point to scale
            xl_origin_new = alb_corr_fit[bands_fit][~bandfit_nan][0:5].mean()
            xr_origin_new = alb_corr_fit[bands_fit][~bandfit_nan][-6:-1].mean()
            xl_fit_new = best_fit_spectrum[bands_fit][~bandfit_nan][0:6].mean()
            xr_fit_new = best_fit_spectrum[bands_fit][~bandfit_nan][-6:-1].mean()
            xfit_base_new = xl_fit_new
            base = np.min([xl_origin_new, xr_origin_new])
            scale = (xr_origin_new - xl_origin_new) / (xr_fit_new - xl_fit_new)
            scale = np.abs(scale)
            replace_array_all = base + (best_fit_spectrum[bands_fit] - xfit_base_new) * scale
            replace_array = replace_array_all[bandfit_nan]
            # print("rescale (2) replace_array_all shape:", replace_array_all.shape)
            # print("rescale replace_array shape:", replace_array.shape)
            # plt.close('all')
            # plt.plot(alb_wvl[bands_fit], alb_corr_mask[bands_fit], 'o', color='k', label='Corrected Albedo')
            # plt.plot(alb_wvl[bands_fit], replace_array_all, '--', color='b', label='Replace All')
            # plt.legend()
            # plt.show()
            # sys.exit()
        elif not np.isfinite(xl_origin) and np.isfinite(xr_origin):
            # only have value on xr_origin
            # not supported yet
            raise NotImplementedError("Only have value on right side is not supported yet.")
        else:
            # print("base, scale:", base, scale)
            # print("base + (best_fit_spectrum[bands_fit][bandfit_nan] - xfit_base) * scale:", replace_array)
            # print("alb_corr_fit[bands_fit][bandfit_nan] after adjustment:", alb_corr_fit[bands_fit][bandfit_nan])
            # plt.close('all')
            # plt.plot(alb_wvl[bands_fit], alb_corr_mask[bands_fit], 'o', color='k', label='Corrected Albedo')
            # plt.plot(alb_wvl[bands_fit], alb_corr_fit_replace, '--', color='b', label='Replace')
            # plt.plot(alb_wvl[bands_fit], alb_corr_fit[bands_fit], '-', color='r', label='Fitted Albedo')
            # plt.xlabel('Wavelength (nm)')
            # plt.ylabel('Albedo')
            # plt.legend()
            # plt.show()
            print("Both sides have no valid value, skip rescaling.")
        
        alb_corr_fit_replace = copy.deepcopy(alb_corr_fit[bands_fit])
        # print('replace_array shape:', replace_array.shape)
        # print('alb_corr_fit_replace shape:', alb_corr_fit_replace.shape)
        # print('bandfit_nan sum:', print(np.sum(bandfit_nan)))
        
        alb_corr_fit_replace[bandfit_nan] = copy.deepcopy(replace_array)
        alb_corr_fit[bands_fit] = copy.deepcopy(alb_corr_fit_replace)
            
        # print("base, scale:", base, scale)
        # print("base + (best_fit_spectrum[bands_fit][bandfit_nan] - xfit_base) * scale:", replace_array)
        # print("alb_corr_fit[bands_fit][bandfit_nan] after adjustment:", alb_corr_fit[bands_fit][bandfit_nan])
        # plt.close('all')
        # plt.plot(alb_wvl[bands_fit], alb_corr_mask[bands_fit], 'o', color='k', label='Corrected Albedo')
        # plt.plot(alb_wvl[bands_fit], alb_corr_fit_replace, '--', color='b', label='Replace')
        # plt.plot(alb_wvl[bands_fit], alb_corr_fit[bands_fit], '-', color='r', label='Fitted Albedo')
        # plt.xlabel('Wavelength (nm)')
        # plt.ylabel('Albedo')
        # plt.legend()
        # plt.show()
            
    
    alb_corr_fit = np.clip(alb_corr_fit, 0, 1)
    
    # plt.close('all')
    # plt.plot(alb_wvl, alb_corr, '-', color='k', label='Corrected Albedo', linewidth=3)
    # plt.plot(alb_wvl, alb_corr_fit, '-', color='r', label='Fitted Albedo', linewidth=1.5)
    # plt.xlabel('Wavelength (nm)')
    # plt.ylabel('Albedo')
    # plt.legend()
    # plt.title(f'SNICAR Best Fit Model: {best_fit_key}, RMSE: {min_rmse:.4f}')
    # plt.show()
    
    # smooth with window size of 5
    alb_corr_fit_smooth = alb_corr_fit.copy()
    alb_corr_fit_smooth = uniform_filter1d(alb_corr_fit_smooth, size=5, mode='reflect')
    alb_corr_fit_smooth = np.clip(alb_corr_fit_smooth, 0, 1)
    
    alb_corr_fit_smooth[np.isfinite(alb_corr_fit_smooth)] = alb_corr_fit_smooth
    
    # print("alb_wvl shape:", alb_wvl.shape)
    # print("alb_corr shape:", alb_corr.shape)
    # print("alb_corr_mask shape:", alb_corr_mask.shape)
    # print("alb_corr_fit shape:", alb_corr_fit.shape)
    # print("alb_corr_fit_smooth shape:", alb_corr_fit_smooth.shape)
    
    return alb_corr_fit_smooth

if __name__ == '__main__':


    pass
