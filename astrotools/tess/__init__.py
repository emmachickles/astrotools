"""TESS full-frame image utilities and forced photometry."""

from .io import (
    download_ffi,
    download_sector_curl_script,
    download_camera_ccd_ffis,
    cleanup_ffi_files,
    load_ffi,
    get_ffi_time,
    iter_ffi_files,
    save_lightcurves,
)
from .lightcurves import find_lightcurves, load_lightcurve
from .matching import match_sources_to_ffi, load_catalog
try:
    from .photometry import aperture_photometry, compute_photometric_scatter
except ImportError:
    pass  # photutils not installed; FFI photometry extraction unavailable

try:
    from . import gpu
except ImportError:
    pass  # GPU dependencies not installed

__all__ = [
    "download_ffi",
    "download_sector_curl_script",
    "download_camera_ccd_ffis",
    "cleanup_ffi_files",
    "load_ffi",
    "get_ffi_time",
    "iter_ffi_files",
    "save_lightcurves",
    "find_lightcurves",
    "load_lightcurve",
    "match_sources_to_ffi",
    "load_catalog",
    "aperture_photometry",
    "compute_photometric_scatter",
]
