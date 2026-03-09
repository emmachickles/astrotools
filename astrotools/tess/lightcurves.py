"""Load TESS forced-photometry lightcurves from numpy arrays by source ID."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import get_tess_dir


def find_lightcurves(source_id, tess_dir=None):
    """Search all id .npy files under tess_dir for a given source.

    Supports two directory layouts:

    * **Engaging format** (``*_id.npy`` / ``*_lc.npy`` / ``*_ts.npy`` /
      ``*_err.npy``) – files share a common prefix stem.
    * **Oreo format** (``id-X-Y.npy`` / ``lc-X-Y.npy`` / ``ts-X-Y.npy``) –
      files share a common numeric suffix ``X-Y``; no separate error array.

    Parameters
    ----------
    source_id : int, float, or str
        Source identifier to search for.  Floats are truncated to int before
        converting to string (e.g. ``2054217744.0`` → ``"2054217744"``).
    tess_dir : str or Path, optional
        Root directory for TESS numpy files.  Defaults to the machine-specific
        path from ``astrotools.config``.

    Returns
    -------
    list of dict
        One entry per file that contains the source, with keys:
        ``ts_path``, ``lc_path``, ``err_path`` (``None`` if absent), ``idx``.
    """
    root = Path(get_tess_dir(tess_dir))

    if isinstance(source_id, float):
        sid = str(int(source_id))
    else:
        sid = str(source_id)

    # Collect id files from both layouts and process each with the right logic.
    # Engaging format: *_id.npy  →  *_lc.npy / *_ts.npy / *_err.npy
    # Oreo format:     id-X-Y.npy  →  lc-X-Y.npy / ts-X-Y.npy (no error)
    all_id_paths = (
        [(p, False) for p in sorted(root.rglob("*_id.npy"))]
        + [(p, True)  for p in sorted(root.rglob("id-*.npy"))]
    )

    results = []
    for id_path, oreo_fmt in all_id_paths:
        ids = np.load(id_path, allow_pickle=True).astype(str)
        matches = np.where(ids == sid)[0]
        if len(matches) == 0:
            continue
        idx = int(matches[0])

        if oreo_fmt:
            # id-X-Y.npy  →  lc-X-Y.npy, ts-X-Y.npy  (no error file)
            name    = id_path.name          # e.g. "id-2-1.npy"
            suffix  = name[len("id"):]      # e.g. "-2-1.npy"
            parent  = id_path.parent
            ts_path  = parent / ("ts" + suffix)
            lc_path  = parent / ("lc" + suffix)
            err_path = None
        else:
            # *_id.npy  →  *_lc.npy, *_ts.npy, *_err.npy
            stem     = str(id_path)[: -len("_id.npy")]
            ts_path  = Path(stem + "_ts.npy")
            lc_path  = Path(stem + "_lc.npy")
            err_path = Path(stem + "_err.npy")
            if not err_path.exists():
                err_path = None

        results.append(
            {
                "ts_path": ts_path,
                "lc_path": lc_path,
                "err_path": err_path,
                "idx": idx,
            }
        )

    return results


def load_lightcurve(source_id, tess_dir=None, normalize=True):
    """Load and concatenate all TESS observations for a source.

    Parameters
    ----------
    source_id : int, float, or str
        Source identifier.
    tess_dir : str or Path, optional
        Root directory for TESS numpy files.
    normalize : bool
        If True, divide flux by the median so the out-of-transit level is ~1.

    Returns
    -------
    dict or None
        Keys: ``time`` (BTJD = BJD − 2 457 000), ``flux``, ``flux_err``
        (``None`` if error arrays are unavailable).
        Returns ``None`` if the source is not found in any file.
    """
    matches = find_lightcurves(source_id, tess_dir=tess_dir)
    if not matches:
        return None

    all_time, all_flux, all_err = [], [], []
    has_err = True

    for m in matches:
        ts = np.load(m["ts_path"])
        lc = np.load(m["lc_path"])[m["idx"]]
        valid = np.isfinite(ts) & np.isfinite(lc)
        if not np.any(valid):
            continue
        all_time.append(ts[valid])
        all_flux.append(lc[valid])
        if m["err_path"] is not None:
            err = np.load(m["err_path"])[m["idx"]][valid]
            all_err.append(err)
        else:
            has_err = False

    if not all_time:
        return None

    time = np.concatenate(all_time)
    flux = np.concatenate(all_flux)
    order = np.argsort(time)
    time = time[order]
    flux = flux[order]

    flux_err = None
    if has_err and all_err:
        flux_err = np.concatenate(all_err)[order]

    if normalize:
        median = np.nanmedian(flux)
        if median != 0 and np.isfinite(median):
            flux = flux / median
            if flux_err is not None:
                flux_err = flux_err / abs(median)

    return {"time": time, "flux": flux, "flux_err": flux_err}
