"""Utilities for matching catalog sources to TESS FFI footprints."""

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord


def match_sources_to_ffi(wcs_obj, catalog_coords, catalog_ids=None, fov_radius=6.0 * u.deg):
    """
    Match catalog sources to a TESS FFI footprint.

    Parameters
    ----------
    wcs_obj : astropy.wcs.WCS
        World coordinate system for the FFI.
    catalog_coords : astropy.coordinates.SkyCoord
        Sky coordinates of catalog sources.
    catalog_ids : numpy.ndarray, optional
        Identifiers for catalog sources. If None, uses indices.
    fov_radius : astropy.units.Quantity
        Field of view radius for initial spatial matching.
        TESS FFI is ~12° diagonal, so sqrt(2)*6° is a safe bound.

    Returns
    -------
    matched_coords : astropy.coordinates.SkyCoord
        Sky coordinates of sources within the FFI footprint.
    matched_ids : numpy.ndarray
        Identifiers of matched sources.
    on_ccd_mask : numpy.ndarray
        Boolean mask of sources that fall on the CCD (0 < x,y < 2048).
    """
    # Get central coordinate of the FFI
    central_coord = SkyCoord.from_pixel(1024, 1024, wcs_obj)

    # Initial spatial cut: sources within field of view
    separations = central_coord.separation(catalog_coords)
    fov_mask = separations < fov_radius

    if not np.any(fov_mask):
        # No sources in field of view
        empty_coords = SkyCoord([], [], unit="deg")
        return empty_coords, np.array([]), np.array([], dtype=bool)

    # Matched sources from spatial cut
    matched_catalog_coords = catalog_coords[fov_mask]
    if catalog_ids is not None:
        matched_catalog_ids = np.asarray(catalog_ids)[fov_mask]
    else:
        matched_catalog_ids = np.arange(len(catalog_coords))[fov_mask]

    # Check which sources fall on the CCD (pixel bounds)
    pixel_coords = matched_catalog_coords.to_pixel(wcs_obj)
    x_pix, y_pix = pixel_coords

    # TESS CCD dimensions: 2048 x 2048
    on_ccd_mask = (
        (x_pix > 0) & (x_pix < 2048) & (y_pix > 0) & (y_pix < 2048)
    )

    # Filter to only on-CCD sources
    final_coords = matched_catalog_coords[on_ccd_mask]
    final_ids = matched_catalog_ids[on_ccd_mask]

    return final_coords, final_ids, on_ccd_mask


def load_catalog(catalog_path, format="csv"):
    """
    Load a white dwarf catalog.

    Parameters
    ----------
    catalog_path : str or Path
        Path to catalog file.
    format : str
        Catalog format: 'csv', 'txt', or 'fits'.

    Returns
    -------
    catalog_coords : astropy.coordinates.SkyCoord
        Sky coordinates of sources.
    catalog_ids : numpy.ndarray
        Source identifiers.
    """
    if format == "csv":
        import pandas as pd

        df = pd.read_csv(catalog_path)

        # Try to auto-detect coordinate columns
        ra_col = None
        dec_col = None
        id_col = None

        for col in df.columns:
            col_lower = col.lower()
            if "ra" in col_lower and ra_col is None:
                ra_col = col
            elif "dec" in col_lower and dec_col is None:
                dec_col = col
            elif "source_id" in col_lower or "id" in col_lower:
                id_col = col

        if ra_col is None or dec_col is None:
            raise ValueError(f"Could not auto-detect RA/Dec columns in {catalog_path}")

        ra = df[ra_col].to_numpy()
        dec = df[dec_col].to_numpy()

        if id_col is not None:
            source_ids = df[id_col].to_numpy()
        else:
            source_ids = np.arange(len(df))

    elif format == "txt":
        # Assume whitespace-separated: ID RA Dec
        data = np.loadtxt(catalog_path, dtype=str, usecols=(0, 1, 2))
        source_ids = data[:, 0]
        ra = data[:, 1].astype(float)
        dec = data[:, 2].astype(float)

    elif format == "fits":
        from astropy.table import Table

        table = Table.read(catalog_path)
        # Auto-detect columns (similar to CSV)
        ra_col = None
        dec_col = None
        id_col = None

        for col in table.colnames:
            col_lower = col.lower()
            if "ra" in col_lower and ra_col is None:
                ra_col = col
            elif "dec" in col_lower and dec_col is None:
                dec_col = col
            elif "source_id" in col_lower or "id" in col_lower:
                id_col = col

        if ra_col is None or dec_col is None:
            raise ValueError(f"Could not auto-detect RA/Dec columns in {catalog_path}")

        ra = table[ra_col]
        dec = table[dec_col]
        source_ids = table[id_col] if id_col else np.arange(len(table))

    else:
        raise ValueError(f"Unsupported catalog format: {format}")

    catalog_coords = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame="icrs")
    return catalog_coords, source_ids


def filter_by_tess_coverage(catalog_coords, catalog_ids, sector=None, camera=None, ccd=None):
    """
    Filter catalog by TESS observational coverage.

    Parameters
    ----------
    catalog_coords : astropy.coordinates.SkyCoord
        Sky coordinates of catalog sources.
    catalog_ids : numpy.ndarray
        Source identifiers.
    sector : int, optional
        TESS sector number. If None, checks for any TESS coverage.
    camera : int, optional
        Camera number (1-4).
    ccd : int, optional
        CCD number (1-4).

    Returns
    -------
    filtered_coords : astropy.coordinates.SkyCoord
        Coordinates of sources with TESS coverage.
    filtered_ids : numpy.ndarray
        Identifiers of filtered sources.

    Notes
    -----
    This function uses tess-point to determine TESS coverage.
    """
    try:
        import tess_stars2px
    except ImportError:
        raise ImportError("tess-point package required for TESS coverage check. Install with: pip install tess-point")

    # Convert to arrays
    ra_deg = catalog_coords.ra.degree
    dec_deg = catalog_coords.dec.degree

    # Query TESS pointings
    if sector is not None:
        # Check specific sector
        outID, outEclipLong, outEclipLat, outSec, outCam, outCcd, \
            outColPix, outRowPix, scinfo = tess_stars2px.tess_stars2px_function_entry(
                0, ra_deg, dec_deg
            )

        mask = (outSec == sector)
        if camera is not None:
            mask &= (outCam == camera)
        if ccd is not None:
            mask &= (outCcd == ccd)

    else:
        # Check for any TESS coverage
        outID, outEclipLong, outEclipLat, outSec, outCam, outCcd, \
            outColPix, outRowPix, scinfo = tess_stars2px.tess_stars2px_function_entry(
                0, ra_deg, dec_deg
            )
        mask = outSec > 0  # Any valid sector

    filtered_coords = catalog_coords[mask]
    filtered_ids = np.asarray(catalog_ids)[mask]

    return filtered_coords, filtered_ids
