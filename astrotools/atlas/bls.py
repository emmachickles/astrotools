"""Helpers for loading ATLAS and TESS BLS search results."""

from __future__ import annotations

import pandas as pd

from .paths import bls_path, tess_bls_path

# EOF_ATLAS format: headerless, 10 columns
ATLAS_COLUMNS = ["gid", "pow", "snr", "wid", "per_day", "per_min", "q", "phi0", "dphi", "epo"]
_ATLAS_NUMERIC = ["pow", "snr", "wid", "per_day", "per_min", "q", "phi0", "dphi", "epo"]

# TESS_Cycle_5 format: comment header, 14 columns
# File header: # ticid, ra, dec, sig, snr, wid, period, period_min, q, phi0, epo, rp, nt, dphi
TESS_COLUMNS = ["ticid", "ra", "dec", "sig", "snr", "wid", "per_day", "per_min", "q", "phi0", "epo", "rp", "nt", "dphi"]
_TESS_NUMERIC = ["ra", "dec", "sig", "snr", "wid", "per_day", "per_min", "q", "phi0", "epo", "rp", "nt", "dphi"]

# Backward-compat alias
COLUMNS = ATLAS_COLUMNS

_atlas_catalog = None
_tess_catalog = None


def load_catalog(bls_dir=None, fmt="atlas"):
    """Load the BLS catalog into a pandas DataFrame.

    Parameters
    ----------
    bls_dir : str or Path, optional
        Override the default directory for this format.
    fmt : {"atlas", "tess"}
        Which result format to load. "atlas" reads EOF_ATLAS-style files
        (10 headerless columns, Gaia IDs); "tess" reads TESS_Cycle_5-style
        files (14 columns with a leading comment header, TIC IDs).
    """

    if fmt == "tess":
        path = tess_bls_path(bls_dir=bls_dir)
        columns = TESS_COLUMNS
        numeric = _TESS_NUMERIC
        index_col = "ticid"
        read_kwargs = dict(names=columns, comment="#")
    else:
        path = bls_path(bls_dir=bls_dir)
        columns = ATLAS_COLUMNS
        numeric = _ATLAS_NUMERIC
        index_col = "gid"
        read_kwargs = dict(names=columns)

    result_files = sorted(path.glob("*.result"))

    dfs = []
    for result_file in result_files:
        df = pd.read_csv(result_file, **read_kwargs)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        empty = pd.DataFrame(columns=columns)
        empty.set_index(index_col, inplace=True)
        return empty

    df = pd.concat(dfs, ignore_index=True)

    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.set_index(index_col, inplace=True)
    return df


def get_catalog(bls_dir=None, fmt="atlas"):
    """Return the cached BLS catalog, loading it on first use.

    Parameters
    ----------
    bls_dir : str or Path, optional
        Override the default directory. Passing this always triggers a reload.
    fmt : {"atlas", "tess"}
        Which result format to load.
    """

    global _atlas_catalog, _tess_catalog

    if fmt == "tess":
        if _tess_catalog is None or bls_dir is not None:
            _tess_catalog = load_catalog(bls_dir=bls_dir, fmt="tess")
        return _tess_catalog
    else:
        if _atlas_catalog is None or bls_dir is not None:
            _atlas_catalog = load_catalog(bls_dir=bls_dir, fmt="atlas")
        return _atlas_catalog


def get_bls_stats(source_id, bls_dir=None, fmt="atlas"):
    """Return the BLS row for a given source id, if present.

    Parameters
    ----------
    source_id : int or float
        Gaia source ID (fmt="atlas") or TIC ID (fmt="tess").
    bls_dir : str or Path, optional
        Override the default directory.
    fmt : {"atlas", "tess"}
        Which result format to query.
    """

    df = get_catalog(bls_dir=bls_dir, fmt=fmt)
    if source_id in df.index:
        return df.loc[source_id]
    return None


def get_period(source_id, bls_dir=None, fmt="atlas"):
    """Return the BLS period in minutes for a given source id, if present.

    Parameters
    ----------
    source_id : int or float
        Gaia source ID (fmt="atlas") or TIC ID (fmt="tess").
    bls_dir : str or Path, optional
        Override the default directory.
    fmt : {"atlas", "tess"}
        Which result format to query.
    """

    df = get_catalog(bls_dir=bls_dir, fmt=fmt)
    if source_id in df.index:
        return df.loc[source_id, "per_min"]
    return None
