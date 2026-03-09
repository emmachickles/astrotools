"""High-level GPU-accelerated TESS sector extraction pipeline."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..io import load_ffi, get_ffi_time, save_lightcurves
from ..matching import match_sources_to_ffi
from ._backend import get_array_module, to_device, to_numpy
from .crowding import compute_contamination_ratios, compute_blending_scores
from .photometry import (
    ApertureIndexSet,
    precompute_pixel_coords,
    precompute_aperture_indices,
    gpu_batch_photometry,
    compute_flux_errors,
    select_best_aperture,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_ffi_batch(ffi_paths, tica=False):
    """Load a batch of FFI images and times from disk.

    Returns
    -------
    images : numpy.ndarray, shape ``(n_frames, ny, nx)``
    times : numpy.ndarray, shape ``(n_frames,)``
    cadences : numpy.ndarray, shape ``(n_frames,)``
    """
    images = []
    times = []
    cadences = []
    for p in ffi_paths:
        img, _, hdr = load_ffi(p, tica=tica)
        t, cad = get_ffi_time(hdr, tica=tica)
        images.append(img.astype(np.float32))
        times.append(t)
        cadences.append(cad)
    return np.stack(images), np.array(times), np.array(cadences)


def _async_load_ffi_batch(executor, ffi_paths, tica=False):
    """Submit a batch load to a thread pool; returns a Future."""
    return executor.submit(_load_ffi_batch, ffi_paths, tica=tica)


def _transfer_index_set(idx_set, xp):
    """Move an ApertureIndexSet to the target device."""
    return ApertureIndexSet(
        indices=to_device(idx_set.indices, xp),
        weights=to_device(idx_set.weights, xp),
        mask=to_device(idx_set.mask, xp),
        area=to_device(idx_set.area, xp),
    )


def _slice_index_set(idx_set, sl):
    """Slice an ApertureIndexSet along the source axis."""
    return ApertureIndexSet(
        idx_set.indices[sl],
        idx_set.weights[sl],
        idx_set.mask[sl],
        idx_set.area[sl],
    )


def _is_oom(exc):
    """Check whether an exception is a GPU out-of-memory error."""
    name = type(exc).__name__
    return "OutOfMemory" in name or "MemoryError" in name


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def extract_sector_gpu(
    sector,
    camera,
    ccd,
    catalog_coords,
    catalog_ids,
    catalog_mags,
    ffi_dir,
    output_dir,
    aperture_radii=None,
    annulus_configs=None,
    contamination_threshold=0.5,
    batch_size=32,
    max_sources_per_chunk=100_000,
    device="gpu",
    bg_method="nanmedian",
    gain=1.0,
    read_noise=10.0,
    contamination_fwhm=1.0,
    contamination_search_radius=10.0,
    save_all_apertures=False,
    tica=False,
    wcs_recompute_interval=None,
):
    """Run GPU-accelerated forced photometry on a full TESS sector CCD.

    Parameters
    ----------
    sector, camera, ccd : int
        TESS sector / camera / CCD identifiers.
    catalog_coords : astropy.coordinates.SkyCoord
        Full catalog sky positions.
    catalog_ids : array-like
        Source identifiers.
    catalog_mags : array-like
        Source magnitudes for contamination analysis.
    ffi_dir : str or Path
        Directory containing FFI FITS files.
    output_dir : str or Path
        Directory for output light-curve files.
    aperture_radii : list of float, optional
        Aperture radii in pixels.  Defaults to ``[0.5, 1.0, 1.5, 2.0, 2.5]``.
    annulus_configs : list of tuple, optional
        ``(r_in, r_out)`` in pixels.
        Defaults to ``[(2.5, 4.0), (3.0, 5.0), (4.0, 6.0)]``.
    contamination_threshold : float
        Sources with contamination above this value are flagged as blended.
    batch_size : int
        Number of FFI frames processed per GPU batch.
    max_sources_per_chunk : int
        Maximum sources processed simultaneously on GPU.
    device : str
        ``"gpu"`` or ``"cpu"``.
    bg_method : str
        Background estimation (``"nanmedian"`` or ``"nanmean"``).
    gain : float
        Detector gain (e-/ADU).
    read_noise : float
        Read noise per pixel (electrons).
    contamination_fwhm : float
        PSF FWHM in pixels for contamination analysis.
    contamination_search_radius : float
        Neighbour search radius in pixels.
    save_all_apertures : bool
        If ``True``, save flux for every aperture config.
    tica : bool
        If ``True``, expect TICA-format FFIs.
    wcs_recompute_interval : int or None
        If set, recompute pixel coordinates from the WCS of every N-th FFI
        to account for pointing drift.  Default ``None`` (compute once from
        the first FFI — adequate for typical TESS pointing stability <0.1 px).

    Returns
    -------
    output_path : Path
        Base path of saved output files.
    """
    ffi_dir = Path(ffi_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if aperture_radii is None:
        aperture_radii = [0.5, 1.0, 1.5, 2.0, 2.5]
    if annulus_configs is None:
        annulus_configs = [(2.5, 4.0), (3.0, 5.0), (4.0, 6.0)]

    xp, actual_device = get_array_module(device)
    log.info("Device: %s (requested %s)", actual_device, device)

    n_configs = len(aperture_radii) * len(annulus_configs)

    # ------------------------------------------------------------------
    # 1. Source matching (CPU) — use first FFI for WCS
    # ------------------------------------------------------------------
    ffi_paths = sorted(ffi_dir.glob("*.fits"))
    if not ffi_paths:
        raise FileNotFoundError(f"No FITS files in {ffi_dir}")

    ref_image, ref_wcs, _ = load_ffi(ffi_paths[0], tica=tica)
    matched_coords, matched_ids, _ = match_sources_to_ffi(
        ref_wcs, catalog_coords, catalog_ids
    )
    n_sources = len(matched_ids)
    log.info("Matched %d sources to CCD", n_sources)
    if n_sources == 0:
        log.warning("No sources on CCD — nothing to extract")
        return None

    # Build magnitude array for matched sources (same order)
    catalog_ids_arr = np.asarray(catalog_ids)
    catalog_mags_arr = np.asarray(catalog_mags, dtype=np.float32)
    matched_ids_arr = np.asarray(matched_ids)
    id_to_idx = {v: i for i, v in enumerate(catalog_ids_arr)}
    matched_mag_indices = np.array([id_to_idx[mid] for mid in matched_ids_arr])
    matched_mags = catalog_mags_arr[matched_mag_indices]

    # ------------------------------------------------------------------
    # 2. Pixel coord precompute (CPU, once)
    # ------------------------------------------------------------------
    x_pix, y_pix = precompute_pixel_coords(ref_wcs, matched_coords)

    # ------------------------------------------------------------------
    # 3. Index precompute (CPU, once) + transfer to device
    # ------------------------------------------------------------------
    image_shape = ref_image.shape
    aperture_sets_cpu, annulus_sets_cpu = precompute_aperture_indices(
        x_pix, y_pix, aperture_radii, annulus_configs, image_shape
    )
    aperture_sets = [_transfer_index_set(s, xp) for s in aperture_sets_cpu]
    annulus_sets = [_transfer_index_set(s, xp) for s in annulus_sets_cpu]

    # ------------------------------------------------------------------
    # 4. Contamination analysis (once, CPU)
    # ------------------------------------------------------------------
    t0 = time.monotonic()
    contam, n_nbrs = compute_contamination_ratios(
        x_pix,
        y_pix,
        matched_mags,
        fwhm_pix=contamination_fwhm,
        search_radius_pix=contamination_search_radius,
        xp=np,
    )
    blended = compute_blending_scores(contam, threshold=contamination_threshold)
    log.info(
        "Contamination computed in %.1fs: median=%.4f, %d/%d flagged (threshold=%.2f)",
        time.monotonic() - t0,
        np.median(contam),
        np.sum(blended),
        n_sources,
        contamination_threshold,
    )

    # ------------------------------------------------------------------
    # 5. Batched FFI extraction
    # ------------------------------------------------------------------
    n_frames_total = len(ffi_paths)
    n_batches = (n_frames_total + batch_size - 1) // batch_size
    npix = image_shape[0] * image_shape[1]

    log.info(
        "%d frames, batch_size=%d (%d batches), %d sources (%d chunks of %d)",
        n_frames_total,
        batch_size,
        n_batches,
        n_sources,
        (n_sources + max_sources_per_chunk - 1) // max_sources_per_chunk,
        max_sources_per_chunk,
    )

    # Accumulators (CPU) — shape (n_configs, n_sources, n_frames_total)
    all_flux = np.zeros((n_configs, n_sources, n_frames_total), dtype=np.float32)
    all_bg = np.zeros((n_configs, n_sources, n_frames_total), dtype=np.float32)
    all_times = np.zeros(n_frames_total, dtype=np.float64)
    all_cadences = np.zeros(n_frames_total, dtype=np.int64)

    frame_offset = 0
    wall_start = time.monotonic()

    with ThreadPoolExecutor(max_workers=1) as executor:
        # Prefetch first batch
        batch_paths = ffi_paths[:batch_size]
        future = _async_load_ffi_batch(executor, batch_paths, tica=tica)

        for b in range(n_batches):
            # Wait for current batch
            images_np, times_np, cadences_np = future.result()
            actual_batch = images_np.shape[0]

            # Prefetch next batch
            next_start = (b + 1) * batch_size
            next_end = min(next_start + batch_size, n_frames_total)
            if next_start < n_frames_total:
                next_paths = ffi_paths[next_start:next_end]
                future = _async_load_ffi_batch(executor, next_paths, tica=tica)

            all_times[frame_offset : frame_offset + actual_batch] = times_np
            all_cadences[frame_offset : frame_offset + actual_batch] = cadences_np

            # Optional WCS recompute for this batch
            if wcs_recompute_interval is not None:
                batch_frame_idx = frame_offset  # index of first frame in batch
                if batch_frame_idx > 0 and batch_frame_idx % wcs_recompute_interval == 0:
                    log.info("Recomputing WCS at frame %d", batch_frame_idx)
                    _, new_wcs, _ = load_ffi(ffi_paths[frame_offset], tica=tica)
                    x_pix, y_pix = precompute_pixel_coords(new_wcs, matched_coords)
                    aperture_sets_cpu, annulus_sets_cpu = precompute_aperture_indices(
                        x_pix, y_pix, aperture_radii, annulus_configs, image_shape
                    )
                    aperture_sets = [_transfer_index_set(s, xp) for s in aperture_sets_cpu]
                    annulus_sets = [_transfer_index_set(s, xp) for s in annulus_sets_cpu]

            # Flatten batch for gather operations
            images_flat_np = images_np.reshape(actual_batch, npix)

            # Transfer to device with OOM retry (halve batch)
            try:
                images_dev = to_device(images_flat_np, xp)
            except Exception as e:
                if _is_oom(e):
                    log.warning(
                        "OOM transferring batch %d (%d frames) — "
                        "processing frame-by-frame on CPU fallback",
                        b, actual_batch,
                    )
                    # Fall through to per-config loop with CPU arrays
                    images_dev = images_flat_np
                    use_xp = np
                else:
                    raise
            else:
                use_xp = xp

            # Choose the right index sets for this batch's device
            if use_xp is np and xp is not np:
                # CPU fallback — use CPU index sets
                batch_ap_sets = aperture_sets_cpu
                batch_ann_sets = annulus_sets_cpu
            else:
                batch_ap_sets = aperture_sets
                batch_ann_sets = annulus_sets

            # Iterate aperture/annulus configs x source chunks
            config_idx = 0
            for ai, ap_set in enumerate(batch_ap_sets):
                for ji, ann_set in enumerate(batch_ann_sets):
                    for chunk_start in range(0, n_sources, max_sources_per_chunk):
                        chunk_end = min(chunk_start + max_sources_per_chunk, n_sources)
                        sl = slice(chunk_start, chunk_end)

                        ap_chunk = _slice_index_set(ap_set, sl)
                        ann_chunk = _slice_index_set(ann_set, sl)

                        try:
                            flux_dev, bg_dev = gpu_batch_photometry(
                                images_dev,
                                ap_chunk,
                                ann_chunk,
                                bg_method=bg_method,
                                xp=use_xp,
                            )
                        except Exception as e:
                            if _is_oom(e):
                                log.warning(
                                    "GPU OOM on config %d chunk %d-%d — CPU fallback",
                                    config_idx, chunk_start, chunk_end,
                                )
                                flux_dev, bg_dev = gpu_batch_photometry(
                                    images_flat_np,
                                    _slice_index_set(aperture_sets_cpu[ai], sl),
                                    _slice_index_set(annulus_sets_cpu[ji], sl),
                                    bg_method=bg_method,
                                    xp=np,
                                )
                            else:
                                raise

                        # flux_dev is (n_frames, n_sources); accumulator is (n_sources, n_frames)
                        all_flux[config_idx, sl, frame_offset : frame_offset + actual_batch] = to_numpy(flux_dev).T
                        all_bg[config_idx, sl, frame_offset : frame_offset + actual_batch] = to_numpy(bg_dev).T

                    config_idx += 1

            frame_offset += actual_batch
            elapsed = time.monotonic() - wall_start
            fps = frame_offset / elapsed if elapsed > 0 else 0
            log.info(
                "Batch %d/%d done (%d frames, %.1f frames/s)",
                b + 1, n_batches, actual_batch, fps,
            )

    # ------------------------------------------------------------------
    # 6. Best aperture selection (CPU)
    # ------------------------------------------------------------------
    best_flux, best_idx = select_best_aperture(all_flux)

    # Error estimation using best config's background (vectorized)
    n_annuli = len(annulus_configs)
    src_idx = np.arange(n_sources)
    best_bg = all_bg[best_idx, src_idx]
    ai_per_source = best_idx // n_annuli
    all_areas = np.stack([to_numpy(ap_set.area) for ap_set in aperture_sets])
    best_area = all_areas[ai_per_source, src_idx]

    flux_err = compute_flux_errors(
        best_area[:, None],
        best_bg,
        best_flux,
        gain=gain,
        read_noise=read_noise,
        xp=np,
    )

    # ------------------------------------------------------------------
    # 7. Save outputs
    # ------------------------------------------------------------------
    coords_arr = np.column_stack(
        [matched_coords.ra.degree, matched_coords.dec.degree]
    )

    stem = f"s{sector:04d}_cam{camera}_ccd{ccd}_gpu"
    output_path = output_dir / stem

    save_lightcurves(
        str(output_path),
        all_times,
        best_flux,
        matched_ids_arr,
        coords_arr,
        cadences=all_cadences,
        flux_errors=flux_err,
    )

    # GPU-pipeline-specific outputs
    np.save(f"{output_path}_contam.npy", contam)
    np.save(f"{output_path}_blend.npy", blended)
    np.save(f"{output_path}_best_ap.npy", best_idx)

    if save_all_apertures:
        np.save(f"{output_path}_lc_all.npy", all_flux)

    total_time = time.monotonic() - wall_start
    log.info(
        "Saved results to %s  (%d sources, %d frames, %.0fs total, %.1f frames/s)",
        output_path,
        n_sources,
        n_frames_total,
        total_time,
        n_frames_total / total_time if total_time > 0 else 0,
    )
    return output_path
