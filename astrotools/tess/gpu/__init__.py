"""GPU-accelerated TESS forced photometry engine.

Provides CuPy-accelerated aperture photometry with automatic NumPy fallback
when no GPU is available.
"""

from ._backend import get_array_module, gpu_available, to_device, to_numpy
from .photometry import (
    ApertureIndexSet,
    precompute_pixel_coords,
    precompute_aperture_indices,
    gpu_aperture_photometry,
    gpu_multi_aperture_photometry,
    gpu_batch_photometry,
    compute_flux_errors,
    select_best_aperture,
)
from .crowding import (
    compute_contamination_ratios,
    compute_blending_scores,
    psf_weighted_extraction,
)
from .pipeline import extract_sector_gpu

__all__ = [
    # backend
    "get_array_module",
    "gpu_available",
    "to_device",
    "to_numpy",
    # photometry
    "ApertureIndexSet",
    "precompute_pixel_coords",
    "precompute_aperture_indices",
    "gpu_aperture_photometry",
    "gpu_multi_aperture_photometry",
    "gpu_batch_photometry",
    "compute_flux_errors",
    "select_best_aperture",
    # crowding
    "compute_contamination_ratios",
    "compute_blending_scores",
    "psf_weighted_extraction",
    # pipeline
    "extract_sector_gpu",
]
