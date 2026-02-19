"""Helpers for loading ATLAS BLS search results."""

from __future__ import annotations

import pandas as pd

from .paths import bls_path

COLUMNS = ["gid", "pow", "snr", "wid", "per_day", "per_min", "q", "phi0", "dphi", "epo"]
_NUMERIC_COLUMNS = ["pow", "snr", "wid", "per_day", "per_min", "q", "phi0", "dphi", "epo"]
_catalog = None


def load_catalog(bls_dir=None):
    """Load the BLS catalog into a pandas DataFrame."""

    result_files = sorted(bls_path(bls_dir=bls_dir).glob("*.result"))

    dfs = []
    for result_file in result_files:
        df = pd.read_csv(result_file, names=COLUMNS)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        empty = pd.DataFrame(columns=COLUMNS)
        empty.set_index("gid", inplace=True)
        return empty

    df = pd.concat(dfs, ignore_index=True)

    for col in _NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["per_day"] = df["per_min"] / 1440
    df.set_index("gid", inplace=True)
    return df


def get_catalog(bls_dir=None):
    """Return the cached BLS catalog, loading it on first use."""

    global _catalog
    if _catalog is None or bls_dir is not None:
        _catalog = load_catalog(bls_dir=bls_dir)
    return _catalog


def get_bls_stats(source_id, bls_dir=None):
    """Return the BLS row for a given source id, if present."""

    df = get_catalog(bls_dir=bls_dir)
    if source_id in df.index:
        return df.loc[source_id]
    return None


def get_period(source_id, bls_dir=None):
    """Return the BLS period (minutes) for a given source id, if present."""

    df = get_catalog(bls_dir=bls_dir)
    if source_id in df.index:
        return df.loc[source_id, "per_min"]
    return None
