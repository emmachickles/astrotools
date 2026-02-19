"""TESS forced photometry utilities."""

import numpy as np
from astropy import units as u
from photutils.aperture import ApertureStats, SkyCircularAperture, SkyCircularAnnulus


def aperture_photometry(
    image,
    wcs,
    sky_coordinates,
    aperture_radius=21.0 * u.arcsec,
    annulus_inner=31.5 * u.arcsec,
    annulus_outer=42.0 * u.arcsec,
):
    """
    Perform aperture photometry on a TESS FFI with background subtraction.

    Parameters
    ----------
    image : numpy.ndarray
        2D calibrated image data.
    wcs : astropy.wcs.WCS
        World coordinate system for the image.
    sky_coordinates : astropy.coordinates.SkyCoord
        Sky coordinates of sources to extract photometry.
    aperture_radius : astropy.units.Quantity
        Radius of the photometric aperture. Default is 21 arcsec (1 TESS pixel).
    annulus_inner : astropy.units.Quantity
        Inner radius of background annulus. Default is 1.5 TESS pixels.
    annulus_outer : astropy.units.Quantity
        Outer radius of background annulus. Default is 2.0 TESS pixels.

    Returns
    -------
    flux : numpy.ndarray
        Background-subtracted flux for each source.
    """
    # Create apertures
    sky_aperture = SkyCircularAperture(sky_coordinates, r=aperture_radius)
    sky_annulus = SkyCircularAnnulus(sky_coordinates, r_in=annulus_inner, r_out=annulus_outer)

    # Convert to pixel coordinates
    pix_aperture = sky_aperture.to_pixel(wcs=wcs)
    pix_annulus = sky_annulus.to_pixel(wcs=wcs)

    # Calculate areas
    sky_area = np.pi * pix_aperture.r**2
    bkg_area = np.pi * (pix_annulus.r_out**2 - pix_annulus.r_in**2)

    # Compute statistics
    sky_stats = ApertureStats(image, pix_aperture)
    bkg_stats = ApertureStats(image, pix_annulus)

    # Background-subtracted flux
    norm = bkg_area / sky_area
    flux = sky_stats.sum - bkg_stats.sum / norm

    return flux


def multi_aperture_photometry(
    image,
    wcs,
    sky_coordinates,
    aperture_radii,
    annulus_configs,
):
    """
    Extract photometry with multiple aperture and annulus configurations.

    Parameters
    ----------
    image : numpy.ndarray
        2D calibrated image data.
    wcs : astropy.wcs.WCS
        World coordinate system for the image.
    sky_coordinates : astropy.coordinates.SkyCoord
        Sky coordinates of sources.
    aperture_radii : list of astropy.units.Quantity
        List of aperture radii to test.
    annulus_configs : list of tuples
        List of (r_in, r_out) annulus configurations.

    Returns
    -------
    flux_array : numpy.ndarray
        Shape (n_configs, n_sources) array of fluxes.
    """
    flux_list = []
    for ap_r in aperture_radii:
        for ann_in, ann_out in annulus_configs:
            flux = aperture_photometry(
                image, wcs, sky_coordinates, aperture_radius=ap_r, annulus_inner=ann_in, annulus_outer=ann_out
            )
            flux_list.append(flux)

    return np.array(flux_list)


def compute_photometric_scatter(flux, method="mad"):
    """
    Compute photometric scatter (precision metric) for a light curve.

    Parameters
    ----------
    flux : numpy.ndarray
        Flux measurements for a single source.
    method : str
        Scatter metric: 'mad' (median absolute deviation), 'std' (standard deviation),
        or 'rms' (root-mean-square).

    Returns
    -------
    scatter : float
        Photometric scatter metric.
    median_flux : float
        Median flux (for normalization or magnitude calculation).
    """
    flux = np.asarray(flux)
    valid = np.isfinite(flux)
    if not np.any(valid):
        return np.nan, np.nan

    flux_valid = flux[valid]
    median_flux = np.median(flux_valid)

    if method == "mad":
        scatter = np.median(np.abs(flux_valid - median_flux))
    elif method == "std":
        scatter = np.std(flux_valid)
    elif method == "rms":
        scatter = np.sqrt(np.mean((flux_valid - median_flux) ** 2))
    else:
        raise ValueError(f"Unknown scatter method: {method}")

    return scatter, median_flux


def flux_to_magnitude(flux, flux_err=None, zero_point=20.44):
    """
    Convert TESS flux to magnitude.

    Parameters
    ----------
    flux : numpy.ndarray or float
        Flux in electrons/s.
    flux_err : numpy.ndarray or float, optional
        Flux uncertainty.
    zero_point : float
        TESS magnitude zero point. Default is 20.44.

    Returns
    -------
    mag : numpy.ndarray or float
        TESS magnitude.
    mag_err : numpy.ndarray or float (if flux_err provided)
        Magnitude uncertainty.
    """
    flux = np.asarray(flux)
    mag = zero_point - 2.5 * np.log10(flux)

    if flux_err is not None:
        flux_err = np.asarray(flux_err)
        mag_err = 2.5 / np.log(10) * flux_err / flux
        return mag, mag_err

    return mag
