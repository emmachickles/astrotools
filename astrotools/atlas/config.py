"""Configuration helpers for locating ATLAS light-curve files."""

from pathlib import Path

# Import from central config module
from ..config import get_atlas_dir


def atlas_base_dir(atlas_dir=None):
    """Resolve the root directory that holds the raw ATLAS light-curve files.

    Resolution order (first match wins):
    1) Explicit ``atlas_dir`` argument.
    2) Hostname-based defaults from central config module.
    
    Parameters
    ----------
    atlas_dir : str or Path, optional
        Explicit path to ATLAS directory. If None, uses hostname-based default.
    
    Returns
    -------
    Path
        Resolved ATLAS directory path.
    """
    return get_atlas_dir(atlas_dir)


def atlas_data_path(*parts, atlas_dir=None):
    """Join paths under the resolved ATLAS light-curve root.
    
    Parameters
    ----------
    *parts : str or Path
        Path components to join under ATLAS root.
    atlas_dir : str or Path, optional
        Explicit path to ATLAS directory. If None, uses hostname-based default.
    
    Returns
    -------
    Path
        Joined path under ATLAS root.
    """
    return atlas_base_dir(atlas_dir).joinpath(*map(Path, parts))


def iter_lightcurve_files(
    atlas_dir=None,
    recursive=True,
    allowed_suffixes=None,
):
    """Yield candidate light-curve files under the ATLAS directory.

    Parameters
    ----------
    atlas_dir : str or Path, optional
        Root directory containing light-curve files. Defaults to resolved base.
    recursive : bool, optional
        If True, recurse through subdirectories; otherwise only top-level files.
    allowed_suffixes : iterable of str, optional
        If provided, only files whose suffix matches one of the supplied values
        will be yielded. Provide values without the leading dot.
    
    Yields
    ------
    Path
        Light curve file paths.
    """
    base = atlas_base_dir(atlas_dir)
    iterator = base.rglob("*") if recursive else base.iterdir()

    normalized_suffixes = None
    if allowed_suffixes is not None:
        normalized_suffixes = {s.lstrip(".") for s in allowed_suffixes}

    for path in iterator:
        if not path.is_file():
            continue
        if normalized_suffixes is not None and path.suffix.lstrip(".") not in normalized_suffixes:
            continue
        yield path
