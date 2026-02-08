"""Light-curve time-series utilities (ported from atlas-quicklook)."""

from __future__ import annotations

import numpy as np
from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord


def bjd_convert(
    time,
    ra: float,
    dec: float,
    date_format: str = "mjd",
    telescope: str = "Palomar",
    scale: str = "tcb",
):
    """Convert times to barycentric Julian dates."""

    coord = SkyCoord(ra, dec, unit="deg")
    time_obj = Time(time, format=date_format, scale="utc")
    time_scaled = time_obj.tcb if scale == "tcb" else time_obj.tdb
    observatory = EarthLocation.of_site(telescope)
    correction = time_scaled.light_travel_time(
        coord, kind="barycentric", location=observatory
    )
    bjd = time_scaled + correction
    return bjd.mjd


def phase_fold(
    time,
    period: float,
    period_derivative: float = 0.0,
    reference_epoch: float | None = None,
):
    """Phase-fold time series data accounting for optional period derivative."""

    time = np.asarray(time)
    if reference_epoch is None:
        reference_epoch = np.min(time)

    dt = time - reference_epoch
    phases = ((dt - 0.5 * period_derivative / period * dt**2) % period) / period
    return phases


def bin_phase_folded_data(
    time,
    flux,
    flux_err,
    period: float,
    period_derivative: float = 0.0,
    reference_epoch: float | None = None,
    num_bins: int = 500,
    num_cycles: int = 3,
    normalization: str | bool = False,
):
    """Bin phase-folded light-curve data with inverse-variance weighting."""

    phases = phase_fold(time, period, period_derivative, reference_epoch)
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    binned = []
    for i, center in enumerate(bin_centers):
        phase_min = bin_edges[i]
        phase_max = bin_edges[i + 1]
        mask = (phases >= phase_min) & (phases < phase_max)
        if not np.any(mask):
            binned.append([center, np.nan, np.nan])
            continue

        bin_flux = np.asarray(flux)[mask]
        bin_flux_err = np.asarray(flux_err)[mask]
        valid = (bin_flux_err > 0) & np.isfinite(bin_flux_err) & np.isfinite(bin_flux)
        if not np.any(valid):
            binned.append([center, np.nan, np.nan])
            continue

        weights = 1.0 / (bin_flux_err[valid] ** 2)
        weighted_mean = np.sum(bin_flux[valid] * weights) / np.sum(weights)
        weighted_error = np.sqrt(1.0 / np.sum(weights))
        binned.append([center, weighted_mean, weighted_error])

    binned = np.array(binned)

    if normalization:
        valid_flux = binned[:, 1][np.isfinite(binned[:, 1])]
        if len(valid_flux) == 0:
            raise ValueError("No valid flux values for normalization")

        norm_methods = {
            "median": np.median,
            "min": np.min,
            "max": np.max,
            "mean": np.mean,
        }
        norm_method = str(normalization).lower()
        if norm_method not in norm_methods:
            raise ValueError(f"Unknown normalization method: {normalization}")
        norm_factor = norm_methods[norm_method](valid_flux)
        binned[:, 1] /= norm_factor
        binned[:, 2] /= norm_factor

    if num_cycles == 1:
        pass
    elif num_cycles == 2:
        prev = binned.copy()
        prev[:, 0] -= 1.0
        binned = np.vstack([prev, binned])
    elif num_cycles == 3:
        prev = binned.copy()
        prev[:, 0] -= 1.0
        nxt = binned.copy()
        nxt[:, 0] += 1.0
        binned = np.vstack([prev, binned, nxt])
    else:
        raise ValueError(f"num_cycles must be 1, 2, or 3, got {num_cycles}")

    return {
        "phase": binned[:, 0],
        "flux": binned[:, 1],
        "flux_err": binned[:, 2],
    }
