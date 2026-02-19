"""Model atmosphere utilities."""

from __future__ import annotations

import os
import re
import warnings

import numpy as np
import astropy.units as u
import astropy.constants as const

DEFAULT_MODEL_DIR = "/data/models/ELM"


def _to_value(value, unit: u.UnitBase):
    if isinstance(value, u.Quantity):
        return value.to_value(unit)
    return np.asarray(value)


def _to_quantity(value, unit: u.UnitBase):
    if isinstance(value, u.Quantity):
        return value.to(unit)
    return np.asarray(value) * unit


def _get_model_dir(data_dir: str | None) -> str:
    if data_dir:
        return data_dir
    return os.environ.get("ASTROTOOLS_MODEL_DIR", DEFAULT_MODEL_DIR)


def _normalize_model_class(model_class: str | None) -> str | None:
    if model_class is None:
        return None
    value = model_class.strip().upper()
    if value in {"ELM", "DA", "H", "HYDROGEN"}:
        return "H"
    if value in {"DB", "HE", "HELIUM"}:
        return "He"
    raise ValueError(
        "Unsupported model_class. Use one of: ELM, DA, DB, H, He."
    )


def _parse_model_filename(fname: str):
    pattern = re.compile(
        r"^NLTE_(?P<comp>[A-Za-z]+)_(?P<teff>\d+\.?\d*)_(?P<g>\d\.\d{3}E[+\-]\d+)_(?P<tag>.+)\.txt$"
    )
    match = pattern.match(fname)
    if not match:
        return None
    comp = match.group("comp")
    teff = float(match.group("teff"))
    gravity = float(match.group("g"))
    return comp, teff, gravity


def _infer_model_composition(parsed_rows):
    compositions = {row[0].upper() for row in parsed_rows}
    if "HE" in compositions:
        return "He"
    return "H"


def load_tremblay_grid(
    data_dir: str | None = None,
    teff_min: float | u.Quantity = 5000 * u.K,
    model_class: str | None = None,
):
    """
    Load Tremblay atmosphere models into a homogeneous grid, suitable for
    interpolation.

    Returns:
    * grid_wavelength : grid of wavelengths (Angstrom)
    * grid_flux : array of shape (len(g_grid), len(teff_grid), len(grid_wavelength))
                  Eddington flux density H_ν in cgs
                  (erg / cm^2 / s / Hz)
    * teff_grid : effective temperatures (K)
    * logg_grid : log10(surface gravities)
    """

    data_dir = _get_model_dir(data_dir)
    fnames = sorted(os.listdir(data_dir))

    parsed = []
    for fname in fnames:
        row = _parse_model_filename(fname)
        if row is not None:
            parsed.append((row[0], row[1], row[2], fname))

    if not parsed:
        raise FileNotFoundError(f"No NLTE model files found in {data_dir}")

    normalized_class = _normalize_model_class(model_class)
    if normalized_class is None:
        normalized_class = _infer_model_composition(parsed)

    parsed = [row for row in parsed if row[0].upper() == normalized_class.upper()]
    if not parsed:
        raise FileNotFoundError(
            f"No NLTE files matching model_class={model_class!r} in {data_dir}"
        )

    g_grid = np.unique([row[2] for row in parsed])
    Teff_grid = np.unique([row[1] for row in parsed])

    teff_min_value = _to_value(teff_min, u.K)
    Teff_grid = Teff_grid[Teff_grid >= teff_min_value]

    file_map = {(row[2], row[1]): row[3] for row in parsed}

    # Load WD atmosphere models for each logG and Teff.
    # Will hold len(g_grid) lists, each with len(Teff_grid) arrays of fluxes.
    modgrid = []
    for gg in g_grid:
        temporaneo = []
        for tt in Teff_grid:
            fname = file_map.get((gg, tt))
            if fname is None:
                raise FileNotFoundError(
                    f"Missing model for Teff={tt}, g={gg} in {data_dir}"
                )
            tempw, tempf = np.loadtxt(
                os.path.join(data_dir, fname),
                unpack=True,
            )
            temporaneo.append(tempf)
        modgrid.append(temporaneo)

    modgrid = np.array(modgrid)
    modwave = tempw

    # Remove models with NaNs.
    inds = np.nonzero(np.isnan(modgrid))
    modgrid = np.delete(modgrid, np.unique(inds[0]), axis=0)
    g_grid = np.delete(g_grid, np.unique(inds[0]))
    modgrid = np.delete(modgrid, np.unique(inds[1]), axis=1)
    Teff_grid = np.delete(Teff_grid, np.unique(inds[1]))

    # Return log of surface gravities.
    lgg_grid = np.log10(g_grid)

    return modwave, modgrid, Teff_grid, lgg_grid


def load_atmosphere_grid(
    model_class: str = "ELM",
    data_dir: str | None = None,
    teff_min: float | u.Quantity = 5000 * u.K,
):
    """Load atmosphere grid by class.

    Supported classes: ``ELM``/``DA`` (hydrogen) and ``DB`` (helium).
    """

    return load_tremblay_grid(
        data_dir=data_dir,
        teff_min=teff_min,
        model_class=model_class,
    )


def interpolate_model_grid(
    wavelength,
    grid_wavelength,
    grid_flux,
    teff_grid,
    logg_grid,
    teff,
    logg,
):
    """Interpolate model atmospheres to requested wavelengths."""

    from scipy.ndimage import map_coordinates

    wave_value = _to_value(wavelength, u.AA)
    modwave_value = _to_value(grid_wavelength, u.AA)
    teff_value = _to_value(teff, u.K)
    logg_value = _to_value(logg, u.dex)

    vectemp = 0 * wave_value + teff_value
    veclogg = 0 * wave_value + logg_value

    # np.interp takes arguments (x, xp, yp).
    teff_grid_value = _to_value(teff_grid, u.K)
    logg_grid_value = _to_value(logg_grid, u.dex)

    windex = np.interp(wave_value, modwave_value, np.arange(len(modwave_value)))
    tindex = np.interp(
        np.log10(vectemp),
        np.log10(teff_grid_value),
        np.arange(len(teff_grid_value)),
    )
    gindex = np.interp(
        np.log10(veclogg),
        np.log10(logg_grid_value),
        np.arange(len(logg_grid_value)),
    )

    flux = map_coordinates(grid_flux, np.array([gindex, tindex, windex]))

    return flux


def interpolate_to_linear_wavelength(
    observed_wavelength,
    grid_wavelength,
    grid_flux,
    teff_grid,
    logg_grid,
    teff,
    logg,
    padding: float | u.Quantity = 10 * u.AA,
    oversample: int = 10,
):
    """Interpolate model atmosphere to an evenly spaced wavelength array."""

    obs_wave = _to_value(observed_wavelength, u.AA)
    mod_wave = _to_value(grid_wavelength, u.AA)
    pad_value = _to_value(padding, u.AA)

    wmin = np.min(obs_wave) - pad_value
    wmax = np.max(obs_wave) + pad_value
    nbin = np.count_nonzero((mod_wave > wmin) * (mod_wave < wmax))
    linwave = np.linspace(wmin, wmax, oversample * nbin)
    modflux = interpolate_model_grid(
        linwave,
        mod_wave,
        grid_flux,
        teff_grid,
        logg_grid,
        teff,
        logg,
    )

    return linwave, modflux


def apply_broadening(
    observed_wavelength,
    linear_wavelength,
    grid_wavelength,
    grid_flux,
    vsini,
    resolution: float = 7000,
    limb_darkening: float = 0.5,
):
    warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"PyAstronomy(\.|$)")
    from PyAstronomy.pyasl import fastRotBroad, instrBroadGaussFast
    from scipy.ndimage import map_coordinates

    # Apply instrumental broadening.
    vsini_value = _to_value(vsini, u.km / u.s)
    resolution_value = float(resolution)

    modflux = instrBroadGaussFast(
        linear_wavelength, grid_flux, resolution_value, edgeHandling="firstlast"
    )

    # Apply rotational broadening.
    modflux = fastRotBroad(linear_wavelength, modflux, limb_darkening, vsini_value)

    # Trim edge effects.
    obs_wave = _to_value(observed_wavelength, u.AA)
    linwave = np.asarray(linear_wavelength)
    inds = np.nonzero((linwave > np.min(obs_wave)) * (linwave < np.max(obs_wave)))
    linwave = linwave[inds]

    # Interpolate model grid to observed wavelength grid.
    windex = np.interp(obs_wave, linwave, np.arange(len(linwave)))
    modflux = map_coordinates(modflux, np.array([windex]))

    return modflux


def fnu_to_flam(wave, flux, return_quantity: bool = False):
    """Convert frequency-space flux density to wavelength-space flux density.

    Notes
    -----
    Tremblay model grids provide Eddington flux H_ν in
    ``erg / cm^2 / s / Hz``. This function applies:

    ``H_λ = H_ν * c / λ^2``
    """

    wave_q = _to_quantity(wave, u.AA)
    flux_q = _to_quantity(flux, u.erg / u.cm**2 / u.Hz / u.s)
    flam = (flux_q * const.c / wave_q**2).to(u.erg / u.cm**2 / u.AA / u.s)
    return flam if return_quantity else flam.value


def scale_model_flux(
    wavelength,
    flux,
    radius,
    distance,
    return_quantity: bool = False,
):
    """Convert Tremblay Eddington flux H_ν to observed Earth flux f_λ.

    The model files store H_ν in ``erg / cm^2 / s / Hz``. The conversion is:

    ``f_ν = 4π (R / D)^2 H_ν``

    followed by ``f_λ = f_ν c / λ^2``.
    """

    wave_q = _to_quantity(wavelength, u.AA)
    flux_q = _to_quantity(flux, u.erg / u.cm**2 / u.Hz / u.s)
    radius_q = _to_quantity(radius, u.R_sun).to(u.cm)
    distance_q = _to_quantity(distance, u.pc).to(u.cm)

    fnu_earth = (4 * np.pi * (radius_q / distance_q) ** 2 * flux_q).to(
        u.erg / u.cm**2 / u.Hz / u.s
    )
    flam_earth = (fnu_earth * const.c / wave_q**2).to(u.erg / u.cm**2 / u.AA / u.s)
    return flam_earth if return_quantity else flam_earth.value


def plot_line_profile_fits(wave, obsflux, modflux, wave0, out_dir, pad=0.3):
    import matplotlib.pyplot as plt
    import numpy as np

    from .spectral_analysis import normalize_continuum

    plt.figure(figsize=(4, 4))
    plt.ylabel("Relative Flux")
    plt.xlabel(r"$\Delta\lambda(Å)$")
    ymin, ymax = 0.0, 1.02 + len(wave) * pad
    plt.yticks(np.arange(ymin, ymax, pad))
    plt.ylim([0.0, plt.ylim()[1]])

    for i, w0 in enumerate(wave0):
        f = normalize_continuum(wave[i], obsflux[i])
        plt.plot(wave[i] - w0, f + pad * i, "-k", lw=0.5)
        f = normalize_continuum(wave[i], modflux[i])
        plt.plot(wave[i] - w0, f + pad * i, "-r", lw=0.5)

    plt.tight_layout()

    fname = os.path.join(out_dir, "line_fits.png")
    plt.savefig(fname, dpi=300)
    print("Saved " + fname)


# Compatibility aliases for legacy imports from SPECTRAL_UTILS.model_utils.
load_models = load_tremblay_grid
interpolate = interpolate_model_grid
interpolate_linear = interpolate_to_linear_wavelength
broadening = apply_broadening
convert_flux_density = fnu_to_flam
convert_to_physical = scale_model_flux
plot_line_fits = plot_line_profile_fits


