"""Utilities for working with ATLAS light curves."""

from __future__ import annotations

from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import atlas_data_path, iter_lightcurve_files
from ..time.series import bjd_convert

# Column indices in the raw ATLAS light-curve files (mirrors atlas-quicklook).
ATLAS_COLS = {
    "mjd": 0,
    "m": 1,
    "dm": 2,
    "ujy": 3,
    "dujy": 4,
    "filter": 5,
    "err": 6,
    "chi_n": 7,
    "ra": 8,
    "dec": 9,
    "x": 10,
    "y": 11,
    "maj": 12,
    "min": 13,
    "phi": 14,
    "apfit": 15,
    "mag5sig": 16,
    "pa_deg": 17,
    "sky": 18,
    "obs": 19,
}

# ── HDF5 index cache ────────────────────────────────────────────────────────

_HDF5_HANDLE = None        # h5py.File (kept open, read-only)
_HDF5_PATH: str | None = None
_HDF5_SOURCE_IDS: np.ndarray | None = None   # sorted int64 array


def _ensure_hdf5_index(hdf5_path):
    """Open the HDF5 file and cache its source_id array for binary search."""
    global _HDF5_HANDLE, _HDF5_PATH, _HDF5_SOURCE_IDS

    path_str = str(hdf5_path)
    if _HDF5_PATH == path_str and _HDF5_HANDLE is not None:
        return True

    if _HDF5_HANDLE is not None:
        try:
            _HDF5_HANDLE.close()
        except Exception:
            pass

    try:
        import h5py
    except ImportError:
        _HDF5_HANDLE = None
        _HDF5_PATH = None
        _HDF5_SOURCE_IDS = None
        return False

    _HDF5_HANDLE = h5py.File(path_str, "r")
    _HDF5_PATH = path_str
    _HDF5_SOURCE_IDS = _HDF5_HANDLE["source_id"][:]
    return True


def _load_lightcurve_hdf5(source_id, hdf5_path):
    """Load a light curve from the ATLAS HDF5 file.

    Returns the same dict format as the text-file path, or None.
    """
    if not _ensure_hdf5_index(hdf5_path):
        return None

    h5 = _HDF5_HANDLE
    ids = _HDF5_SOURCE_IDS

    idx = int(np.searchsorted(ids, source_id))
    if idx >= len(ids) or ids[idx] != source_id:
        return None

    lo = int(h5["offsets"][idx])
    hi = int(h5["offsets"][idx + 1])
    if hi <= lo:
        return None

    ra = float(h5["ra"][idx])
    dec = float(h5["dec"][idx])

    t_mjd = h5["mjd"][lo:hi].astype(np.float64)
    t_mid_mjd = t_mjd + 15.0 / 86400.0   # midpoint of 30 s exposure
    t_bjd = bjd_convert(t_mid_mjd, ra, dec, date_format="mjd")

    flux = h5["ujy"][lo:hi].astype(np.float64)
    flux_err = h5["dujy"][lo:hi].astype(np.float64)
    mag = h5["mag"][lo:hi].astype(np.float64)
    mag_err = h5["dmag"][lo:hi].astype(np.float64)

    filt_uint8 = h5["filt"][lo:hi]
    filter_col = np.where(filt_uint8 == 1, "o", "c")

    return {
        "time": t_bjd,
        "flux": flux,
        "flux_err": flux_err,
        "mag": mag,
        "mag_err": mag_err,
        "filter": filter_col,
        "ra": ra,
        "dec": dec,
        "path": Path(str(source_id)),
    }


# ── public helpers ───────────────────────────────────────────────────────────

def lightcurve_path(source, atlas_dir=None):
    """Return the file path for a given source identifier or path-like."""

    if isinstance(source, Path):
        return source
    return atlas_data_path(str(source), atlas_dir=atlas_dir)


def list_lightcurve_files(
    atlas_dir=None,
    recursive=True,
    allowed_suffixes=None,
    limit=None,
):
    """Collect light-curve file paths under the ATLAS directory."""

    files = []
    for idx, path in enumerate(
        iter_lightcurve_files(
            atlas_dir=atlas_dir, recursive=recursive, allowed_suffixes=allowed_suffixes
        )
    ):
        files.append(path)
        if limit is not None and idx + 1 >= limit:
            break
    return files


def count_lightcurves(
    atlas_dir=None,
    recursive=True,
    allowed_suffixes=None,
):
    """Return the number of available ATLAS light-curve files."""

    return sum(
        1
        for _ in iter_lightcurve_files(
            atlas_dir=atlas_dir, recursive=recursive, allowed_suffixes=allowed_suffixes
        )
    )


def choose_random_lightcurve(
    atlas_dir=None,
    recursive=True,
    allowed_suffixes=None,
    seed=None,
):
    """Randomly select a light-curve file path."""

    files = list_lightcurve_files(
        atlas_dir=atlas_dir, recursive=recursive, allowed_suffixes=allowed_suffixes
    )
    if not files:
        raise FileNotFoundError(
            "No ATLAS light-curve files found. Check atlas_dir or ATLAS_LC_DIR."
        )

    rng = random.Random(seed)
    return rng.choice(files)


def load_lightcurve(source, atlas_dir=None):
    """Load a single ATLAS light curve into numpy arrays.

    If *source* is a string or integer Gaia ID and an ATLAS HDF5 file is
    available, the light curve is read from HDF5 (fast).  Otherwise (or if the
    source is not found in HDF5) the original text-file path is used.

    Returns a dictionary with ``time``, ``flux``, ``flux_err``, ``filter``,
    ``ra``, ``dec``, or ``None`` on failure.
    """

    # ── try HDF5 for non-Path sources ────────────────────────────────────
    if not isinstance(source, Path):
        try:
            source_id = int(source)
        except (ValueError, TypeError):
            source_id = None

        if source_id is not None:
            from ..config import get_atlas_hdf5

            hdf5_path = get_atlas_hdf5()
            if hdf5_path is not None:
                result = _load_lightcurve_hdf5(source_id, hdf5_path)
                if result is not None:
                    return result

    # ── text-file fallback ───────────────────────────────────────────────
    path = lightcurve_path(source, atlas_dir=atlas_dir)
    try:
        df = pd.read_csv(path, sep=r"\s+", header=None, comment="#")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return None

    ra = float(df.iloc[0, ATLAS_COLS["ra"]])
    dec = float(df.iloc[0, ATLAS_COLS["dec"]])

    t_mjd = df.iloc[:, ATLAS_COLS["mjd"]].to_numpy(float)
    t_mid_mjd = t_mjd + 15.0 / 86400.0  # midpoint of 30s exposure
    t_bjd = bjd_convert(t_mid_mjd, ra, dec, date_format="mjd")

    flux = df.iloc[:, ATLAS_COLS["ujy"]].to_numpy(float)
    flux_err = df.iloc[:, ATLAS_COLS["dujy"]].to_numpy(float)
    filter_col = df.iloc[:, ATLAS_COLS["filter"]].to_numpy(str)
    mag = df.iloc[:, ATLAS_COLS["m"]].to_numpy(float)
    mag_err = df.iloc[:, ATLAS_COLS["dm"]].to_numpy(float)

    return {
        "time": t_bjd,
        "flux": flux,
        "flux_err": flux_err,
        "mag": mag,
        "mag_err": mag_err,
        "filter": filter_col,
        "ra": ra,
        "dec": dec,
        "path": Path(path),
    }


def plot_lightcurve(
    lc_data,
    source_id=None,
    figsize=(10, 4),
    dpi=120,
    alpha=0.7,
    marker_size=2.5,
):
    """Plot a light curve separated by ATLAS filters."""

    if lc_data is None:
        raise ValueError("lc_data is None; did the load fail?")

    times = lc_data["time"]
    flux = lc_data["flux"]
    flux_err = lc_data["flux_err"]
    filters = lc_data["filter"]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    colors = {"o": "orange", "c": "cyan"}

    for filt in sorted(set(filters)):
        mask = filters == filt
        ax.errorbar(
            times[mask],
            flux[mask],
            flux_err[mask],
            fmt=".",
            ms=marker_size,
            alpha=alpha,
            capsize=3,
            label=f"Filter {filt}",
            color=colors.get(filt, "gray"),
        )

    ax.set_xlabel("Time (BJD)")
    ax.set_ylabel("Flux (uJy)")
    if source_id is None and "path" in lc_data:
        source_id = Path(lc_data["path"]).name
    if source_id:
        ax.set_title(f"Gaia DR3 ID: {source_id}")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig
