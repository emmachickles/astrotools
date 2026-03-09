"""GPU-accelerated forced aperture photometry for TESS FFIs."""

from collections import namedtuple

import numpy as np

from ._backend import get_array_module, to_device, to_numpy

ApertureIndexSet = namedtuple(
    "ApertureIndexSet", ["indices", "weights", "mask", "area"]
)
"""Precomputed aperture gather indices for all sources.

Fields
------
indices : int32 array, shape ``(n_sources, max_pix)``
    Flat pixel indices into the image.
weights : float32 array, shape ``(n_sources, max_pix)``
    Partial pixel coverage weights.
mask : bool array, shape ``(n_sources, max_pix)``
    Valid pixel flag (``False`` for out-of-bounds or padding).
area : float32 array, shape ``(n_sources,)``
    Effective aperture area (sum of weights per source).
"""


# ---------------------------------------------------------------------------
# Coordinate precomputation
# ---------------------------------------------------------------------------

def precompute_pixel_coords(wcs_obj, sky_coordinates):
    """Convert sky coordinates to pixel coordinates once per sector.

    Parameters
    ----------
    wcs_obj : astropy.wcs.WCS
        WCS from a representative FFI.
    sky_coordinates : astropy.coordinates.SkyCoord
        Catalog sky positions.

    Returns
    -------
    x_pix, y_pix : numpy.ndarray
        Float32 pixel coordinates.
    """
    x, y = sky_coordinates.to_pixel(wcs_obj)
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)


# ---------------------------------------------------------------------------
# Aperture index precomputation (fully vectorized)
# ---------------------------------------------------------------------------

def _build_disk_offsets(radius):
    """Enumerate integer (dy, dx) offsets within a circular aperture.

    Returns offsets where ``dy**2 + dx**2 <= (radius + 0.5)**2`` to capture
    all pixels whose centres might be within the aperture.
    """
    r_search = int(np.ceil(radius + 0.5))
    ys = np.arange(-r_search, r_search + 1)
    xs = np.arange(-r_search, r_search + 1)
    dy_grid, dx_grid = np.meshgrid(ys, xs, indexing="ij")
    dist_sq = dy_grid ** 2 + dx_grid ** 2
    inside = dist_sq <= (radius + 0.5) ** 2
    return dy_grid[inside].astype(np.int32), dx_grid[inside].astype(np.int32)


def _build_annulus_offsets(r_inner, r_outer):
    """Enumerate integer (dy, dx) offsets within an annulus."""
    r_search = int(np.ceil(r_outer + 0.5))
    ys = np.arange(-r_search, r_search + 1)
    xs = np.arange(-r_search, r_search + 1)
    dy_grid, dx_grid = np.meshgrid(ys, xs, indexing="ij")
    dist_sq = dy_grid ** 2 + dx_grid ** 2
    inside = (dist_sq <= (r_outer + 0.5) ** 2) & (dist_sq >= (r_inner - 0.5) ** 2)
    return dy_grid[inside].astype(np.int32), dx_grid[inside].astype(np.int32)


def precompute_aperture_indices(
    x_pix,
    y_pix,
    aperture_radii_pix,
    annulus_configs_pix,
    image_shape=(2048, 2048),
):
    """Precompute flat pixel indices and weights for all sources and configs.

    Fully vectorized — no per-source Python loops.

    Parameters
    ----------
    x_pix, y_pix : numpy.ndarray
        Float32 pixel coordinates of sources, shape ``(n_sources,)``.
    aperture_radii_pix : list of float
        Aperture radii in pixels (e.g. ``[1.0, 1.5, 2.0]``).
    annulus_configs_pix : list of tuple
        ``(r_inner, r_outer)`` annulus configurations in pixels.
    image_shape : tuple of int
        ``(ny, nx)`` image dimensions.

    Returns
    -------
    aperture_sets : list of ApertureIndexSet
        One per aperture radius.
    annulus_sets : list of ApertureIndexSet
        One per annulus configuration.
    """
    ny, nx = image_shape
    n_sources = len(x_pix)

    cx_round = np.round(x_pix).astype(np.int32)  # (n_sources,)
    cy_round = np.round(y_pix).astype(np.int32)
    cx_frac = x_pix - cx_round  # fractional offset from pixel centre
    cy_frac = y_pix - cy_round

    # -- Apertures --
    aperture_sets = []
    for radius in aperture_radii_pix:
        dy_off, dx_off = _build_disk_offsets(radius)  # (max_pix,)
        max_pix = len(dy_off)

        # Broadcast: (n_sources, 1) + (1, max_pix) → (n_sources, max_pix)
        rows = cy_round[:, None] + dy_off[None, :]
        cols = cx_round[:, None] + dx_off[None, :]

        in_bounds = (rows >= 0) & (rows < ny) & (cols >= 0) & (cols < nx)

        flat = rows * nx + cols
        flat = np.where(in_bounds, flat, 0)  # safe dummy for out-of-bounds

        # Per-pixel weights based on distance from fractional source centre
        dist = np.sqrt(
            (dy_off[None, :] - cy_frac[:, None]) ** 2
            + (dx_off[None, :] - cx_frac[:, None]) ** 2
        )
        w = np.clip(radius - dist + 0.5, 0.0, 1.0).astype(np.float32)

        mask = in_bounds & (w > 0)
        area = np.sum(w * mask, axis=1).astype(np.float32)

        aperture_sets.append(
            ApertureIndexSet(flat.astype(np.int32), w, mask, area)
        )

    # -- Annuli --
    annulus_sets = []
    for r_inner, r_outer in annulus_configs_pix:
        dy_off, dx_off = _build_annulus_offsets(r_inner, r_outer)
        max_pix = len(dy_off)

        rows = cy_round[:, None] + dy_off[None, :]
        cols = cx_round[:, None] + dx_off[None, :]

        in_bounds = (rows >= 0) & (rows < ny) & (cols >= 0) & (cols < nx)

        flat = rows * nx + cols
        flat = np.where(in_bounds, flat, 0)

        weights = np.ones((n_sources, max_pix), dtype=np.float32)
        area = np.sum(in_bounds.astype(np.float32), axis=1).astype(np.float32)

        annulus_sets.append(
            ApertureIndexSet(flat.astype(np.int32), weights, in_bounds, area)
        )

    return aperture_sets, annulus_sets


# ---------------------------------------------------------------------------
# Single-frame photometry
# ---------------------------------------------------------------------------

def gpu_aperture_photometry(
    image_flat,
    ap_indices,
    ap_weights,
    ap_mask,
    ap_area,
    ann_indices,
    ann_mask,
    bg_method="nanmedian",
    xp=np,
):
    """Extract background-subtracted aperture photometry for one frame.

    All array inputs must already live on the target device (CPU or GPU).

    Parameters
    ----------
    image_flat : 1-D array, shape ``(npix,)``
        Flattened image.
    ap_indices, ap_weights, ap_mask : arrays, shape ``(n_sources, max_ap_pix)``
        Aperture gather indices, weights, and validity mask.
    ap_area : 1-D array, shape ``(n_sources,)``
        Effective aperture area.
    ann_indices, ann_mask : arrays, shape ``(n_sources, max_ann_pix)``
        Annulus gather indices and validity mask.
    bg_method : str
        ``"nanmedian"`` (default, robust) or ``"nanmean"``.
    xp : module
        ``numpy`` or ``cupy``.

    Returns
    -------
    flux : 1-D array, shape ``(n_sources,)``
        Background-subtracted aperture flux.
    bg_per_pix : 1-D array, shape ``(n_sources,)``
        Background level per pixel.
    """
    # Gather aperture pixels
    ap_vals = image_flat[ap_indices]  # (n_sources, max_ap_pix)
    ap_flux = xp.sum(ap_vals * ap_weights * ap_mask, axis=1)

    # Gather annulus pixels and compute background
    ann_vals = image_flat[ann_indices]  # (n_sources, max_ann_pix)
    ann_vals_masked = xp.where(ann_mask, ann_vals, xp.nan)

    if bg_method == "nanmedian":
        bg_per_pix = xp.nanmedian(ann_vals_masked, axis=1)
    elif bg_method == "nanmean":
        bg_per_pix = xp.nanmean(ann_vals_masked, axis=1)
    else:
        raise ValueError(f"Unknown bg_method: {bg_method}")

    # Replace NaN backgrounds with 0 (fully masked sources)
    bg_per_pix = xp.where(xp.isnan(bg_per_pix), 0.0, bg_per_pix)

    flux = ap_flux - bg_per_pix * ap_area
    return flux, bg_per_pix


def gpu_multi_aperture_photometry(
    image_flat,
    aperture_sets,
    annulus_sets,
    bg_method="nanmedian",
    xp=np,
):
    """Extract photometry for all aperture/annulus configurations on one frame.

    Parameters
    ----------
    image_flat : 1-D array, shape ``(npix,)``
    aperture_sets : list of transferred ApertureIndexSet
    annulus_sets : list of transferred ApertureIndexSet
    bg_method : str
    xp : module

    Returns
    -------
    flux_all : 2-D array, shape ``(n_configs, n_sources)``
    bg_all : 2-D array, shape ``(n_configs, n_sources)``
    """
    flux_list = []
    bg_list = []
    for ap_set in aperture_sets:
        for ann_set in annulus_sets:
            flux, bg = gpu_aperture_photometry(
                image_flat,
                ap_set.indices,
                ap_set.weights,
                ap_set.mask,
                ap_set.area,
                ann_set.indices,
                ann_set.mask,
                bg_method=bg_method,
                xp=xp,
            )
            flux_list.append(flux)
            bg_list.append(bg)
    return xp.stack(flux_list), xp.stack(bg_list)


# ---------------------------------------------------------------------------
# Batch (multi-frame) photometry
# ---------------------------------------------------------------------------

def gpu_batch_photometry(
    images_flat,
    ap_set,
    ann_set,
    bg_method="nanmedian",
    xp=np,
):
    """Extract photometry for a batch of FFI frames with one aperture config.

    Aperture sums are fully vectorised over frames.  Annulus medians are
    computed per-frame to limit peak memory usage.

    Parameters
    ----------
    images_flat : 2-D array, shape ``(n_frames, npix)``
        Batch of flattened images on device.
    ap_set : ApertureIndexSet (on device)
    ann_set : ApertureIndexSet (on device)
    bg_method : str
    xp : module

    Returns
    -------
    flux : 2-D array, shape ``(n_frames, n_sources)``
    bg : 2-D array, shape ``(n_frames, n_sources)``
    """
    n_frames = images_flat.shape[0]
    n_sources = ap_set.indices.shape[0]

    # Batched aperture gather: (n_frames, n_sources, max_ap_pix)
    ap_vals = images_flat[:, ap_set.indices]
    ap_flux = xp.sum(
        ap_vals * ap_set.weights[None, :, :] * ap_set.mask[None, :, :],
        axis=2,
    )  # (n_frames, n_sources)

    # Annulus background
    if bg_method == "nanmean":
        # Fully vectorizable — no per-frame loop needed
        ann_vals = images_flat[:, ann_set.indices]  # (n_frames, n_sources, max_ann_pix)
        ann_masked = xp.where(ann_set.mask[None, :, :], ann_vals, xp.nan)
        bg_arr = xp.nanmean(ann_masked, axis=2)
    else:
        # nanmedian — per-frame to limit memory (sorting is expensive)
        bg_arr = xp.empty((n_frames, n_sources), dtype=xp.float32)
        for f in range(n_frames):
            ann_vals = images_flat[f][ann_set.indices]  # (n_sources, max_ann_pix)
            ann_masked = xp.where(ann_set.mask, ann_vals, xp.nan)
            bg_arr[f] = xp.nanmedian(ann_masked, axis=1)

    bg_arr = xp.where(xp.isnan(bg_arr), 0.0, bg_arr)
    flux = ap_flux - bg_arr * ap_set.area[None, :]
    return flux, bg_arr


# ---------------------------------------------------------------------------
# Error estimation
# ---------------------------------------------------------------------------

def compute_flux_errors(ap_area, bg_per_pix, ap_flux, gain=1.0, read_noise=10.0, xp=np):
    """Estimate flux uncertainties from Poisson + background noise.

    Parameters
    ----------
    ap_area : array, shape ``(..., n_sources)``
        Effective aperture area in pixels.
    bg_per_pix : array, shape ``(..., n_sources)``
        Background per pixel.
    ap_flux : array, shape ``(..., n_sources)``
        Background-subtracted source flux (electrons or e-/s).
    gain : float
        Detector gain (e-/ADU). For calibrated TESS images this is ~1.
    read_noise : float
        Read noise per pixel in electrons.
    xp : module
        ``numpy`` or ``cupy``.

    Returns
    -------
    flux_err : array, same shape as *ap_flux*
        1-sigma flux uncertainty.
    """
    source_var = xp.abs(ap_flux) / gain
    bg_var = ap_area * (xp.abs(bg_per_pix) / gain + read_noise ** 2)
    flux_err = xp.sqrt(source_var + bg_var)
    return flux_err


# ---------------------------------------------------------------------------
# Best aperture selection (vectorized)
# ---------------------------------------------------------------------------

def select_best_aperture(flux_all, method="min_scatter"):
    """Select the optimal aperture configuration per source.

    Parameters
    ----------
    flux_all : numpy.ndarray, shape ``(n_configs, n_sources, n_frames)``
        Flux time-series for each aperture configuration.
    method : str
        ``"min_scatter"`` — pick the config with lowest MAD / median.

    Returns
    -------
    best_flux : numpy.ndarray, shape ``(n_sources, n_frames)``
        Flux from the best aperture for each source.
    best_idx : numpy.ndarray, shape ``(n_sources,)``
        Index of the best aperture config per source.
    """
    if method != "min_scatter":
        raise ValueError(f"Unknown selection method: {method}")

    n_configs, n_sources, n_frames = flux_all.shape

    # Vectorized scatter: MAD / |median| for each (config, source)
    n_valid = np.sum(np.isfinite(flux_all), axis=2)  # (n_configs, n_sources)
    med = np.nanmedian(flux_all, axis=2)  # (n_configs, n_sources)
    mad = np.nanmedian(
        np.abs(flux_all - med[:, :, None]), axis=2
    )  # (n_configs, n_sources)

    # Scatter = MAD / |median|; inf where insufficient data or zero median
    with np.errstate(divide="ignore", invalid="ignore"):
        scatter = np.where(
            (n_valid >= 3) & (med != 0),
            mad / np.abs(med),
            np.inf,
        )

    best_idx = np.argmin(scatter, axis=0)  # (n_sources,)
    best_flux = flux_all[best_idx, np.arange(n_sources)]  # (n_sources, n_frames)

    return best_flux, best_idx
