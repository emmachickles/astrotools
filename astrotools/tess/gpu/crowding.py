"""Blending and contamination analysis for crowded TESS fields."""

import numpy as np
from scipy.spatial import cKDTree

from ._backend import to_device, to_numpy


def compute_contamination_ratios(
    x_pix,
    y_pix,
    magnitudes,
    fwhm_pix=1.0,
    search_radius_pix=10.0,
    xp=np,
):
    """Estimate flux contamination from neighboring sources.

    Uses a KD-tree for O(N log N) neighbor search, then vectorized
    sparse operations to compute PSF-weighted contamination.

    Parameters
    ----------
    x_pix, y_pix : 1-D array, shape ``(n_sources,)``
        Pixel positions of all sources.
    magnitudes : 1-D array, shape ``(n_sources,)``
        Catalog magnitudes (e.g. Gaia G or TESS mag).
    fwhm_pix : float
        PSF FWHM in pixels.  TESS undersampled ~ 1 pixel.
    search_radius_pix : float
        Maximum distance (pixels) to consider neighbours.
    xp : module
        ``numpy`` or ``cupy`` (computation done on CPU, results
        transferred to device if needed).

    Returns
    -------
    contamination : 1-D array, shape ``(n_sources,)``
        Fractional contamination ``[0, 1)``.  Zero means no neighbours
        contribute flux; close to 1 means the source is heavily blended.
    n_neighbors : 1-D array of int, shape ``(n_sources,)``
        Number of significant neighbours per source.
    """
    x_np = np.asarray(x_pix, dtype=np.float64)
    y_np = np.asarray(y_pix, dtype=np.float64)
    mag_np = np.asarray(magnitudes, dtype=np.float32)
    n_sources = len(x_np)

    sigma = fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma2 = sigma ** 2

    # Relative fluxes (arbitrary reference mag = 15)
    rel_flux = (10.0 ** (-0.4 * (mag_np - 15.0))).astype(np.float64)

    # KD-tree neighbor search
    coords = np.column_stack([x_np, y_np])
    tree = cKDTree(coords)
    dist_matrix = tree.sparse_distance_matrix(
        tree, search_radius_pix, output_type="coo_matrix"
    )

    # Remove self-pairs (diagonal)
    not_self = dist_matrix.row != dist_matrix.col
    src_idx = dist_matrix.row[not_self]
    nbr_idx = dist_matrix.col[not_self]
    dists = dist_matrix.data[not_self]

    # PSF-weighted flux contributions
    psf_weights = np.exp(-0.5 * dists ** 2 / sigma2)
    contributions = rel_flux[nbr_idx] * psf_weights

    # Sum neighbor flux per source
    neighbor_flux = np.zeros(n_sources, dtype=np.float64)
    np.add.at(neighbor_flux, src_idx, contributions)

    # Count neighbors per source
    n_neighbors = np.zeros(n_sources, dtype=np.int32)
    np.add.at(n_neighbors, src_idx, 1)

    # Contamination ratio
    total = rel_flux + neighbor_flux
    contamination = np.where(
        total > 0, neighbor_flux / total, 0.0
    ).astype(np.float32)

    if xp is not np:
        return to_device(contamination, xp), to_device(n_neighbors, xp)
    return contamination, n_neighbors


def compute_blending_scores(contamination, threshold=0.1):
    """Flag sources above a contamination threshold.

    Parameters
    ----------
    contamination : array, shape ``(n_sources,)``
        Contamination ratios from :func:`compute_contamination_ratios`.
    threshold : float
        Contamination level above which a source is flagged.

    Returns
    -------
    blended : bool array, shape ``(n_sources,)``
        ``True`` for sources whose contamination exceeds *threshold*.
    """
    contamination = np.asarray(to_numpy(contamination))
    return contamination > threshold


def psf_weighted_extraction():
    """PSF-weighted deblending extraction.

    .. note::
        Not yet implemented — placeholder for future PSF-fitting photometry
        that deblends overlapping sources.

    Algorithm sketch (for future implementation):

    1. Build a PSF model per source (Gaussian with FWHM from PRF, or
       empirical PRF from TESS SPOC).
    2. For each group of overlapping sources, construct a design matrix
       ``A`` where column *j* is the PSF of source *j* evaluated at the
       pixel positions of the aperture region.
    3. Solve ``A @ f = d`` (least-squares) for the flux vector ``f``,
       where ``d`` is the data in the aperture pixels.
    4. Iterate groups across the CCD using the same spatial grid as
       :func:`compute_contamination_ratios`.

    Raises
    ------
    NotImplementedError
    """
    raise NotImplementedError(
        "PSF-weighted extraction is a future enhancement. "
        "Use aperture photometry with contamination flagging for now."
    )
