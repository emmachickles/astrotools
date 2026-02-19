"""Spectral analysis utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import astropy.units as u
import astropy.constants as const

from ..time.barycentric import bjd_time as BJDConvert


def _to_value(value, unit: u.UnitBase):
    if isinstance(value, u.Quantity):
        return value.to_value(unit)
    return np.asarray(value)


def _to_quantity(value, unit: u.UnitBase):
    if isinstance(value, u.Quantity):
        return value.to(unit)
    return np.asarray(value) * unit


def doppler_shift_wavelength(wavelength, radial_velocity, return_quantity: bool = False):
    """Shift wavelengths by a radial velocity."""

    wave_q = _to_quantity(wavelength, u.AA)
    rv_q = _to_quantity(radial_velocity, u.km / u.s)
    shifted = wave_q * (1 + rv_q / const.c)
    return shifted if return_quantity else shifted.to_value(u.AA)


def sigma_clip_outliers(wave, flux, ferr=None, window_fraction: float = 0.05, sigma: float = 2.0):
    """Clip non-physical jumps in flux using a rolling standard deviation."""

    import pandas as pd

    window = max(1, int(len(flux) * window_fraction))
    rolling_std = pd.Series(flux).rolling(window=window).std()
    threshold = sigma * np.nanmedian(rolling_std)
    inds = np.nonzero(rolling_std < threshold)

    wave, flux = wave[inds], flux[inds]
    if ferr is None:
        return wave, flux
    ferr = ferr[inds]
    return wave, flux, ferr


def linear_continuum(wave, slope, intercept):
    wave_value = _to_value(wave, u.AA)
    return slope * wave_value + intercept


def voigt_line_with_continuum(wave, x_0, amp_L, fwhm_L, fwhm_G, slope, intercept):
    from astropy.modeling.models import Voigt1D

    wave_value = _to_value(wave, u.AA)
    model = Voigt1D(x_0, amp_L, fwhm_L, fwhm_G)(wave_value)
    continuum = linear_continuum(wave_value, slope, intercept)
    return model + continuum


def multi_voigt_lines(wave, wave0, rv, slope, intercept, amp_L, fwhm_L, fwhm_G):
    """
    Expects wave to be a list, with len(wave) = len(wave0)
    """

    from astropy.modeling.models import Voigt1D

    c = const.c.to_value(u.km / u.s)
    rv_value = _to_value(rv, u.km / u.s)
    flux = []
    for i in range(len(wave0)):
        wave_value = _to_value(wave[i], u.AA)
        wshift = rv_value * wave_value / c
        x_0 = _to_value(wave0[i], u.AA) - wshift
        flux.append(
            voigt_line_with_continuum(
                wave_value,
                x_0,
                amp_L,
                fwhm_L,
                fwhm_G,
                slope,
                intercept,
            )
        )
    return np.array(flux)


def normalize_continuum(wave, flux, ferr=None, deg=1, pad=20, wave0=None):
    wave_value = _to_value(wave, u.AA)
    if wave0 is None:
        wave0 = np.median(wave_value)
    else:
        wave0 = _to_value(wave0, u.AA)

    # Extract continuum region without absorption line.
    pad_value = _to_value(pad, u.AA)
    inds = np.nonzero(np.abs(wave_value - wave0) > pad_value)
    wave_cont, flux_cont = wave_value[inds], flux[inds]
    if ferr is not None:
        ferr_cont = ferr[inds]
    else:
        ferr_cont = None

    # Fit continuum.
    p = np.polyfit(wave_cont, flux_cont, deg)

    # Normalize.
    continuum = np.poly1d(p)(wave_value)
    flux = flux / continuum

    if ferr is None:
        return flux
    ferr = ferr / continuum
    return flux, ferr


def extract_line_windows(
    wave,
    flux,
    ferr,
    wave0,
    half_width: float | u.Quantity = 100 * u.AA,
    sigclip: bool = False,
    norm: bool = False,
):

    wave_line, flux_line, ferr_line = [], [], []
    wave_value = _to_value(wave, u.AA)
    wave0_value = _to_value(wave0, u.AA)

    for w0 in wave0_value:
        # Define region around absorption line.
        pad_value = _to_value(half_width, u.AA)
        wmin = w0 - pad_value
        wmax = w0 + pad_value
        inds = np.nonzero((wave_value > wmin) * (wave_value < wmax))

        # Extract region.
        wave_clip = wave_value[inds]
        flux_clip = flux[inds]
        ferr_clip = ferr[inds]

        # Remove outliers.
        if sigclip:
            wave_clip, flux_clip, ferr_clip = sigma_clip_outliers(
                wave_clip,
                flux_clip,
                ferr_clip,
            )

        if norm:
            # Normalize region.
            flux_clip, ferr_clip = normalize_continuum(wave_clip, flux_clip, ferr_clip)

        # Append to list.
        wave_line.append(wave_clip)
        flux_line.append(flux_clip)
        ferr_line.append(ferr_clip)

    return wave_line, flux_line, ferr_line


def read_hst_stis_spectrum(fits_path, row: int = 0):
    """Load a 1D STIS spectrum from a HLSP/MAST ``cspec`` or ``aspec`` FITS file."""

    from astropy.io import fits

    fits_path = Path(fits_path)
    with fits.open(fits_path) as hdul:
        sci = hdul[1].data
        wave = np.asarray(sci["WAVELENGTH"][row], dtype=float)
        flux = np.asarray(sci["FLUX"][row], dtype=float)
        ferr = np.asarray(sci["ERROR"][row], dtype=float)
        header = hdul[0].header.copy()

    return wave, flux, ferr, header


def _gaussian_broaden_constant_resolution(wave, flux, resolution: float):
    """Approximate instrumental broadening with a Gaussian kernel at constant R."""

    from scipy.ndimage import gaussian_filter1d

    wave = np.asarray(wave)
    flux = np.asarray(flux)

    if resolution is None or resolution <= 0:
        return flux.copy()

    dlam = np.nanmedian(np.diff(wave))
    lam = np.nanmedian(wave)
    fwhm = lam / float(resolution)
    sigma_pix = (fwhm / 2.355) / dlam
    if not np.isfinite(sigma_pix) or sigma_pix <= 0:
        return flux.copy()

    return gaussian_filter1d(flux, sigma=sigma_pix, mode="nearest")


def estimate_resolution_from_wavelength_grid(
    wavelength,
    pixels_per_resolution_element: float = 2.0,
):
    """Estimate resolving power R from sampling in an observed wavelength grid.

    This uses a median-sampling proxy:
    ``R_sampling = median(lambda / delta_lambda)`` and
    ``R_inst ≈ R_sampling / pixels_per_resolution_element``.
    """

    wave = np.asarray(wavelength, dtype=float)
    wave = wave[np.isfinite(wave)]
    if wave.size < 3:
        raise ValueError("Need at least 3 wavelength points to estimate resolution.")

    wave = np.sort(wave)
    dlam = np.diff(wave)
    good = np.isfinite(dlam) & (dlam > 0)
    if not np.any(good):
        raise ValueError("Could not estimate wavelength spacing from input grid.")

    dlam_med = np.median(dlam[good])
    lam_med = np.median(wave)
    r_sampling = lam_med / dlam_med
    r_inst = r_sampling / float(pixels_per_resolution_element)
    return float(r_inst)


def _broaden_model_for_fit(
    observed_wavelength,
    model_wavelength,
    model_grid,
    teff_grid,
    logg_grid,
    teff,
    logg,
    resolution,
    vsini,
    limb_darkening,
):
    from .model_atmospheres import (
        apply_broadening,
        interpolate_model_grid,
        interpolate_to_linear_wavelength,
    )

    obs_wave = np.asarray(observed_wavelength, dtype=float)

    if (vsini is None) or (float(vsini) <= 0):
        model_flux = interpolate_model_grid(
            obs_wave,
            model_wavelength,
            model_grid,
            teff_grid,
            logg_grid,
            teff,
            logg,
        )
        return _gaussian_broaden_constant_resolution(obs_wave, model_flux, resolution)

    linear_wave, linear_flux = interpolate_to_linear_wavelength(
        obs_wave,
        model_wavelength,
        model_grid,
        teff_grid,
        logg_grid,
        teff,
        logg,
    )
    return apply_broadening(
        observed_wavelength=obs_wave,
        linear_wavelength=linear_wave,
        grid_wavelength=model_wavelength,
        grid_flux=linear_flux,
        vsini=vsini,
        resolution=resolution,
        limb_darkening=limb_darkening,
    )


def _fit_scale_and_offset(data, model, error):
    """Weighted fit of ``data ≈ a * model + b``."""

    weights = 1.0 / np.square(error)
    matrix = np.vstack([model, np.ones_like(model)]).T
    sqrtw = np.sqrt(weights)
    aw = matrix * sqrtw[:, None]
    yw = data * sqrtw
    coeff, *_ = np.linalg.lstsq(aw, yw, rcond=None)
    scale, offset = coeff
    fitted = scale * model + offset
    chi2 = np.sum(np.square((data - fitted) / error))
    return scale, offset, chi2


def fit_teff_logg_grid(
    wave,
    flux,
    ferr,
    model_wavelength,
    model_grid,
    teff_grid,
    logg_grid,
    teff_range=None,
    logg_range=None,
    wave_range=(1150.0, 1700.0),
    exclude_ranges=((1214.5, 1216.8),),
    resolution: float | None = 1200.0,
    vsini: float | None = 0.0,
    limb_darkening: float = 0.5,
    refine_continuous: bool = True,
):
    """Brute-force grid fit in ``Teff`` and ``logg`` for UV spectra."""

    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    ferr = np.asarray(ferr, dtype=float)

    mask = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(ferr) & (ferr > 0)
    if wave_range is not None:
        mask &= (wave >= wave_range[0]) & (wave <= wave_range[1])
    if exclude_ranges is not None:
        for lo, hi in exclude_ranges:
            mask &= ~((wave >= lo) & (wave <= hi))

    fit_wave = wave[mask]
    fit_flux = flux[mask]
    fit_ferr = ferr[mask]

    teff_values = np.asarray(teff_grid, dtype=float)
    logg_values = np.asarray(logg_grid, dtype=float)

    if teff_range is not None:
        teff_values = teff_values[(teff_values >= teff_range[0]) & (teff_values <= teff_range[1])]
    if logg_range is not None:
        logg_values = logg_values[(logg_values >= logg_range[0]) & (logg_values <= logg_range[1])]

    if len(fit_wave) < 10:
        raise ValueError("Too few data points remain after filtering.")
    if len(teff_values) == 0 or len(logg_values) == 0:
        raise ValueError("No Teff/logg grid points left after range filters.")

    n_dof = len(fit_wave) - 2
    best = None
    grid_rows = []

    for teff in teff_values:
        for logg in logg_values:
            model_flux = _broaden_model_for_fit(
                observed_wavelength=fit_wave,
                model_wavelength=model_wavelength,
                model_grid=model_grid,
                teff_grid=teff_grid,
                logg_grid=logg_grid,
                teff=teff,
                logg=logg,
                resolution=resolution,
                vsini=vsini,
                limb_darkening=limb_darkening,
            )
            scale, offset, chi2 = _fit_scale_and_offset(fit_flux, model_flux, fit_ferr)
            redchi2 = chi2 / max(1, n_dof)
            row = (teff, logg, scale, offset, chi2, redchi2)
            grid_rows.append(row)

            if (best is None) or (chi2 < best["chi2"]):
                best = {
                    "teff": float(teff),
                    "logg": float(logg),
                    "scale": float(scale),
                    "offset": float(offset),
                    "chi2": float(chi2),
                    "redchi2": float(redchi2),
                    "n_points": int(len(fit_wave)),
                }

    if refine_continuous and best is not None:
        try:
            from scipy.optimize import minimize

            teff_min, teff_max = float(np.min(teff_values)), float(np.max(teff_values))
            logg_min, logg_max = float(np.min(logg_values)), float(np.max(logg_values))

            def objective(params):
                teff_trial, logg_trial = params
                if not (teff_min <= teff_trial <= teff_max and logg_min <= logg_trial <= logg_max):
                    return np.inf
                model_flux = _broaden_model_for_fit(
                    observed_wavelength=fit_wave,
                    model_wavelength=model_wavelength,
                    model_grid=model_grid,
                    teff_grid=teff_grid,
                    logg_grid=logg_grid,
                    teff=teff_trial,
                    logg=logg_trial,
                    resolution=resolution,
                    vsini=vsini,
                    limb_darkening=limb_darkening,
                )
                _, _, chi2_trial = _fit_scale_and_offset(fit_flux, model_flux, fit_ferr)
                return chi2_trial

            x0 = np.array([best["teff"], best["logg"]], dtype=float)
            result = minimize(
                objective,
                x0,
                method="L-BFGS-B",
                bounds=[(teff_min, teff_max), (logg_min, logg_max)],
            )

            if result.success and np.isfinite(result.fun):
                teff_refined, logg_refined = result.x
                model_flux = _broaden_model_for_fit(
                    observed_wavelength=fit_wave,
                    model_wavelength=model_wavelength,
                    model_grid=model_grid,
                    teff_grid=teff_grid,
                    logg_grid=logg_grid,
                    teff=teff_refined,
                    logg=logg_refined,
                    resolution=resolution,
                    vsini=vsini,
                    limb_darkening=limb_darkening,
                )
                scale, offset, chi2 = _fit_scale_and_offset(fit_flux, model_flux, fit_ferr)
                redchi2 = chi2 / max(1, n_dof)

                if chi2 < best["chi2"]:
                    best = {
                        "teff": float(teff_refined),
                        "logg": float(logg_refined),
                        "scale": float(scale),
                        "offset": float(offset),
                        "chi2": float(chi2),
                        "redchi2": float(redchi2),
                        "n_points": int(len(fit_wave)),
                        "refined": True,
                    }
                else:
                    best["refined"] = False
            else:
                best["refined"] = False
        except Exception:
            best["refined"] = False

    return {
        "best": best,
        "grid": np.array(grid_rows, dtype=float),
        "fit_wave": fit_wave,
        "fit_flux": fit_flux,
        "fit_ferr": fit_ferr,
    }


def model_flux_for_solution(
    wave,
    model_wavelength,
    model_grid,
    teff_grid,
    logg_grid,
    teff,
    logg,
    scale: float = 1.0,
    offset: float = 0.0,
    resolution: float | None = 1200.0,
    vsini: float | None = 0.0,
    limb_darkening: float = 0.5,
):
    """Evaluate the model spectrum at a best-fit solution."""

    model_flux = _broaden_model_for_fit(
        observed_wavelength=wave,
        model_wavelength=model_wavelength,
        model_grid=model_grid,
        teff_grid=teff_grid,
        logg_grid=logg_grid,
        teff=teff,
        logg=logg,
        resolution=resolution,
        vsini=vsini,
        limb_darkening=limb_darkening,
    )
    return scale * model_flux + offset


# Compatibility aliases for legacy imports from SPECTRAL_UTILS.spec_utils.
shift_wavelength = doppler_shift_wavelength
clip_outliers = sigma_clip_outliers
linear_model = linear_continuum
line_model = voigt_line_with_continuum
extract_lines = extract_line_windows



