"""Binary-system helper functions for SED and RV analyses."""

from __future__ import annotations

import numpy as np
import astropy.constants as const
import astropy.units as u


def _to_value(value, unit: u.UnitBase):
    if isinstance(value, u.Quantity):
        return value.to_value(unit)
    return np.asarray(value)


def _to_quantity(value, unit: u.UnitBase):
    if isinstance(value, u.Quantity):
        return value.to(unit)
    return np.asarray(value) * unit


def inclination_from_cosi(cos_inclination, return_quantity: bool = False):
    """Convert cos(i) to inclination angle."""

    cosi = np.asarray(cos_inclination)
    inclination = np.arccos(cosi) * u.rad
    if return_quantity:
        return inclination
    return inclination.to_value(u.rad)


def roche_lobe_radius(
    mass_primary,
    mass_secondary,
    orbital_period,
    return_quantity: bool = False,
):
    """Donor Roche-lobe radius from Eggleton (1983).

    Returns radius in solar radii by default.
    """

    m1 = _to_quantity(mass_primary, u.M_sun)
    m2 = _to_quantity(mass_secondary, u.M_sun)
    porb = _to_quantity(orbital_period, u.s)

    q = (m2 / m1).to_value(u.dimensionless_unscaled)
    r_l_over_a = 0.49 * q ** (2 / 3) / (0.6 * q ** (2 / 3) + np.log(1 + q ** (1 / 3)))

    m_tot = m1 + m2
    semi_major_axis = (const.G * m_tot * porb**2 / (4 * np.pi**2)) ** (1 / 3)
    r_l = r_l_over_a * semi_major_axis.to(u.R_sun)

    if return_quantity:
        return r_l
    return r_l.to_value(u.R_sun)


def secondary_rv_semiamplitude(
    cos_inclination,
    mass_primary,
    mass_secondary,
    orbital_period,
    return_quantity: bool = False,
):
    """RV semi-amplitude of the secondary (K2)."""

    cosi = np.asarray(cos_inclination)
    m1 = _to_quantity(mass_primary, u.M_sun)
    m2 = _to_quantity(mass_secondary, u.M_sun)
    porb = _to_quantity(orbital_period, u.s)

    q = (m2 / m1).to_value(u.dimensionless_unscaled)
    m_tot = m1 + m2
    semi_major_axis = (const.G * m_tot * porb**2 / (4 * np.pi**2)) ** (1 / 3)
    vscale = (2 * np.pi * semi_major_axis / porb).to(u.km / u.s)
    inclination = np.arccos(cosi)
    k2 = vscale / (1 + q) * np.sin(inclination)

    if return_quantity:
        return k2
    return k2.to_value(u.km / u.s)


def log_surface_gravity(radius, mass, cgs: bool = True):
    """Compute log10(surface gravity) with stellar mass/radius inputs."""

    radius_q = _to_quantity(radius, u.R_sun)
    mass_q = _to_quantity(mass, u.M_sun)
    gravity = const.G * mass_q / radius_q**2
    if cgs:
        gravity = gravity.to(u.cm / u.s**2)
    return np.log10(gravity.value)


def gaussian_log_probability(parameter, mean, sigma):
    """Gaussian log-probability term (without normalization constant)."""

    return -0.5 * ((np.asarray(parameter) - np.asarray(mean)) ** 2) / np.asarray(sigma) ** 2


# Compatibility aliases for existing scripts.
calculate_radius = roche_lobe_radius
calculate_rv = secondary_rv_semiamplitude
calculate_logg = log_surface_gravity
gauss_prob = gaussian_log_probability
