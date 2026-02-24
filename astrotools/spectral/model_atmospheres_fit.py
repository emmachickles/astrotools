import numpy as np
from scipy.optimize import curve_fit
from .model_atmospheres import interpolate_model_grid, fnu_to_flam

def fit_spectrum_curve_fit(
    wave_obs,
    flux_obs,
    ferr_obs,
    model_wavelength,
    model_grid,
    teff_grid,
    logg_grid,
    p0=(30000, 8.0),
    bounds=((10000, 6.0), (100000, 9.5)),
    resolution=None,
    vsini=None,
    limb_darkening=None,
):
    """
    Fit Teff and logg to observed spectrum using scipy.optimize.curve_fit.
    Returns best-fit parameters and model.
    """
    def model_func(wave, teff, logg):
        model_hnu = interpolate_model_grid(
            wave,
            model_wavelength,
            model_grid,
            teff_grid,
            logg_grid,
            teff,
            logg,
        )
        model_hlam = fnu_to_flam(wave, model_hnu)
        # Fit scale factor analytically for each (teff, logg)
        denom = np.sum((model_hlam / ferr_obs) ** 2)
        if denom <= 0:
            return np.full_like(wave, np.nan)
        scale = np.sum(flux_obs * model_hlam / (ferr_obs ** 2)) / denom
        return scale * model_hlam

    popt, pcov = curve_fit(
        model_func,
        wave_obs,
        flux_obs,
        sigma=ferr_obs,
        p0=p0,
        bounds=bounds,
        absolute_sigma=True,
        maxfev=10000,
    )
    best_teff, best_logg = popt
    best_model = model_func(wave_obs, best_teff, best_logg)
    chi2 = np.sum(((flux_obs - best_model) / ferr_obs) ** 2)
    dof = max(1, len(wave_obs) - len(popt))
    redchi2 = chi2 / dof
    return {
        'teff': float(best_teff),
        'logg': float(best_logg),
        'model_fit': best_model,
        'chi2': float(chi2),
        'dof': dof,
        'redchi2': float(redchi2),
        'pcov': pcov,
        'popt': popt,
    }
