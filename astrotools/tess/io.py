"""TESS FFI I/O utilities."""

import os
import subprocess
from pathlib import Path

import numpy as np
from astropy import wcs
from astropy.io import fits


def download_sector_curl_script(sector, output_dir=None, script_type="ffic"):
    """
    Download the MAST bulk download curl script for a TESS sector.

    Parameters
    ----------
    sector : int
        TESS sector number.
    output_dir : str or Path, optional
        Directory to save the script. If None, uses current directory.
    script_type : str
        Type of script: 'ffic' (FFI calibrated), 'ffi' (FFI raw), etc.

    Returns
    -------
    script_path : Path
        Path to the downloaded script.
    """
    if output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # MAST archive URL pattern
    url = f"https://archive.stsci.edu/missions/tess/download_scripts/sector/tesscurl_sector_{sector}_{script_type}.sh"
    script_name = f"tesscurl_sector_{sector}_{script_type}.sh"
    script_path = output_dir / script_name
    
    # Download using curl
    cmd = ["curl", "-f", "-L", "-o", str(script_path), url]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Downloaded {script_name} ({script_path.stat().st_size / 1024 / 1024:.1f} MB)")
        return script_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to download curl script for sector {sector}: {e.stderr}")


def download_camera_ccd_ffis(
    sector,
    camera,
    ccd,
    output_dir,
    curl_script=None,
    limit=None,
    cleanup_script=True,
):
    """
    Download TESS FFIs for a specific camera/CCD using MAST bulk download scripts.

    Parameters
    ----------
    sector : int
        TESS sector number.
    camera : int
        Camera number (1-4).
    ccd : int
        CCD number (1-4).
    output_dir : str or Path
        Base output directory. FFIs will be saved to output_dir/cam{camera}-ccd{ccd}/
    curl_script : str or Path, optional
        Path to existing curl script. If None, downloads it automatically.
    limit : int, optional
        Maximum number of FFIs to download. If None, downloads all.
    cleanup_script : bool
        If True, removes temporary download script after use.

    Returns
    -------
    ffi_dir : Path
        Directory containing downloaded FFI files.
    n_downloaded : int
        Number of files downloaded.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download curl script if not provided
    if curl_script is None:
        print(f"Downloading bulk download script for sector {sector}...")
        curl_script = download_sector_curl_script(sector, output_dir)
        downloaded_script = True
    else:
        curl_script = Path(curl_script)
        downloaded_script = False
    
    if not curl_script.exists():
        raise FileNotFoundError(f"Curl script not found: {curl_script}")
    
    # Create camera/CCD output directory
    ffi_dir = output_dir / f"cam{camera}-ccd{ccd}"
    ffi_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract lines for this camera/CCD
    pattern = f"s{sector:04d}-{camera}-{ccd}-"
    print(f"Extracting download commands for camera {camera}, CCD {ccd}...")
    
    with open(curl_script, "r") as f:
        lines = [line for line in f if pattern in line]
    
    if limit is not None:
        lines = lines[:limit]
        print(f"Will download {len(lines)} FFI files (limited to {limit})")
    else:
        print(f"Will download {len(lines)} FFI files")
    
    if not lines:
        print(f"No FFI files found for camera {camera}, CCD {ccd}")
        return ffi_dir, 0
    
    # Create temporary download script
    temp_script = ffi_dir / "download_subset.sh"
    with open(temp_script, "w") as f:
        f.write("#!/bin/bash\n")
        for line in lines:
            f.write(line)
    
    temp_script.chmod(0o755)
    
    # Run download
    print(f"Downloading to {ffi_dir}...")
    try:
        subprocess.run(
            ["bash", str(temp_script)],
            cwd=ffi_dir,
            check=True,
            capture_output=False,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Download failed: {e}")
    finally:
        # Cleanup temporary script
        if temp_script.exists():
            temp_script.unlink()
    
    # Count downloaded files
    ffi_files = list(ffi_dir.glob("*.fits"))
    n_downloaded = len(ffi_files)
    
    # Cleanup downloaded curl script if requested
    if cleanup_script and downloaded_script and curl_script.exists():
        curl_script.unlink()
    
    print(f"Downloaded {n_downloaded} FFI files to {ffi_dir}")
    return ffi_dir, n_downloaded


def cleanup_ffi_files(ffi_dir, keep_script=False):
    """
    Remove FITS files from an FFI directory to save space.

    Parameters
    ----------
    ffi_dir : str or Path
        Directory containing FFI FITS files.
    keep_script : bool
        If True, keeps any .sh script files.

    Returns
    -------
    n_removed : int
        Number of files removed.
    space_freed_mb : float
        Approximate space freed in MB.
    """
    ffi_dir = Path(ffi_dir)
    if not ffi_dir.exists():
        return 0, 0.0
    
    ffi_files = list(ffi_dir.glob("*.fits"))
    total_size = sum(f.stat().st_size for f in ffi_files)
    
    for f in ffi_files:
        f.unlink()
    
    n_removed = len(ffi_files)
    space_freed_mb = total_size / (1024 * 1024)
    
    print(f"Removed {n_removed} FITS files, freed {space_freed_mb:.1f} MB")
    return n_removed, space_freed_mb


def download_ffi(sector, camera, ccd, orbit, output_dir, tica=True):
    """
    Download a single TESS FFI using curl.

    Parameters
    ----------
    sector : int
        TESS sector number.
    camera : int
        Camera number (1-4).
    ccd : int
        CCD number (1-4).
    orbit : str
        Orbit identifier (e.g., 'o1a', 'o1b', 'o2a', 'o2b').
    output_dir : str or Path
        Directory to save the FFI.
    tica : bool
        If True, download TICA FFIs. If False, download SPOC FFIs.

    Returns
    -------
    success : bool
        True if download succeeded.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if tica:
        # TICA FFI download
        base_url = "https://mast.stsci.edu/api/v0.1/Download/file/?uri=mast:HLSP/tica"
        sector_str = f"s{sector:04d}"
        cam_ccd_str = f"cam{camera}-ccd{ccd}"
        # Example filename pattern (may need adjustment based on actual TICA naming)
        # This is a placeholder - actual implementation would query MAST or use manifest
        print(f"TICA download for sector {sector}, cam {camera}, ccd {ccd}, orbit {orbit}")
        print("Note: Implement actual TICA download logic based on MAST archive structure")
        return False
    else:
        # SPOC FFI download
        print(f"SPOC download for sector {sector}, cam {camera}, ccd {ccd}")
        print("Note: Implement actual SPOC download logic")
        return False


def download_sector_ffis(sector, camera, ccd, output_dir, tica=True):
    """
    Download all FFIs for a given sector/camera/CCD.

    Parameters
    ----------
    sector : int
        TESS sector number.
    camera : int
        Camera number (1-4).
    ccd : int
        CCD number (1-4).
    output_dir : str or Path
        Directory to save FFIs.
    tica : bool
        If True, download TICA FFIs. If False, download SPOC FFIs.

    Returns
    -------
    ffi_paths : list of Path
        Paths to downloaded FFI files.
    """
    output_dir = Path(output_dir)
    ccd_dir = output_dir / f"cam{camera}-ccd{ccd}"
    ccd_dir.mkdir(parents=True, exist_ok=True)

    # Download across all orbits
    orbits = ["o1a", "o1b", "o2a", "o2b"]
    for orbit in orbits:
        download_ffi(sector, camera, ccd, orbit, ccd_dir, tica=tica)

    # Return list of downloaded files
    if ccd_dir.exists():
        return sorted(ccd_dir.glob("*.fits"))
    return []


def check_ffi_integrity(ffi_dir, min_size_mb=1.0):
    """
    Check integrity of downloaded FFIs by file size.

    Parameters
    ----------
    ffi_dir : str or Path
        Directory containing FFI files.
    min_size_mb : float
        Minimum file size in MB to be considered valid.

    Returns
    -------
    valid_files : list of Path
        List of valid FFI file paths.
    invalid_files : list of Path
        List of potentially corrupted files.
    """
    ffi_dir = Path(ffi_dir)
    min_size_bytes = min_size_mb * 1024 * 1024

    valid = []
    invalid = []

    for fpath in sorted(ffi_dir.glob("*.fits")):
        if fpath.stat().st_size >= min_size_bytes:
            valid.append(fpath)
        else:
            invalid.append(fpath)

    return valid, invalid


def load_ffi(ffi_path, calibrated=True, tica=False):
    """
    Load a TESS FFI and extract image data and WCS.

    Parameters
    ----------
    ffi_path : str or Path
        Path to TESS FFI FITS file.
    calibrated : bool
        If True, load calibrated image (extension 1 for SPOC, extension 0 for TICA).
        If False, load raw image (extension 0).
    tica : bool
        If True, expect TICA format. If False, expect SPOC format.

    Returns
    -------
    image : numpy.ndarray
        2D image array.
    wcs_obj : astropy.wcs.WCS
        World coordinate system.
    header : astropy.io.fits.Header
        Primary header.
    """
    with fits.open(ffi_path) as hdul:
        primary_header = hdul[0].header

        if tica:
            # TICA format: calibrated data in extension 0
            image = hdul[0].data
            image_header = hdul[0].header
        else:
            # SPOC format: calibrated data in extension 1
            if calibrated:
                image = hdul[1].data
                image_header = hdul[1].header
            else:
                image = hdul[0].data
                image_header = hdul[0].header

        wcs_obj = wcs.WCS(image_header)

    return image, wcs_obj, primary_header


def get_ffi_time(header, tica=False):
    """
    Extract time information from TESS FFI header.

    Parameters
    ----------
    header : astropy.io.fits.Header
        FITS header from FFI.
    tica : bool
        If True, expect TICA format (STARTTJD). If False, expect SPOC format (TSTART).

    Returns
    -------
    time : float
        Observation time in appropriate format (TJD for TICA, BTJD for SPOC).
    cadence : int
        Cadence number or FFI index.
    """
    if tica:
        time = header.get("STARTTJD", np.nan)  # TJD (JD - 2457000)
        cadence = header.get("CADENCE", -1)
    else:
        time = header.get("TSTART", np.nan)  # BTJD (BJD - 2457000)
        cadence = header.get("FFIINDEX", -1)

    return time, cadence


def iter_ffi_files(ffi_dir, pattern="*.fits"):
    """
    Iterate over FFI files in a directory.

    Parameters
    ----------
    ffi_dir : str or Path
        Directory containing FFI files.
    pattern : str
        Glob pattern for FFI files.

    Yields
    ------
    ffi_path : Path
        Path to each FFI file.
    """
    ffi_dir = Path(ffi_dir)
    for fpath in sorted(ffi_dir.glob(pattern)):
        yield fpath


def save_lightcurves(
    output_path,
    times,
    fluxes,
    source_ids,
    coordinates,
    cadences=None,
    flux_errors=None,
):
    """
    Save extracted light curves to numpy format.

    Parameters
    ----------
    output_path : str or Path
        Output file path (without extension).
    times : numpy.ndarray
        1D array of observation times.
    fluxes : numpy.ndarray
        2D array of fluxes, shape (n_sources, n_times).
    source_ids : numpy.ndarray
        1D array of source identifiers.
    coordinates : numpy.ndarray
        2D array of coordinates (RA, Dec), shape (n_sources, 2).
    cadences : numpy.ndarray, optional
        1D array of cadence numbers.
    flux_errors : numpy.ndarray, optional
        2D array of flux uncertainties, same shape as fluxes.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort by time
    time_order = np.argsort(times)
    times_sorted = times[time_order]
    fluxes_sorted = fluxes[:, time_order]

    # Save arrays
    np.save(f"{output_path}_ts.npy", times_sorted)
    np.save(f"{output_path}_lc.npy", fluxes_sorted)
    np.save(f"{output_path}_id.npy", source_ids)
    np.save(f"{output_path}_co.npy", coordinates)

    if cadences is not None:
        cadences_sorted = cadences[time_order]
        np.save(f"{output_path}_cn.npy", cadences_sorted)

    if flux_errors is not None:
        flux_errors_sorted = flux_errors[:, time_order]
        np.save(f"{output_path}_err.npy", flux_errors_sorted)
