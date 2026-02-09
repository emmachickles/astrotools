"""Configuration helpers for locating ATLAS light-curve files."""

from pathlib import Path
import socket

# Host-specific defaults mirroring the atlas-quicklook project.
_DEFAULT_ATLAS_HOST_PATHS = {
    "hypernova": Path("/data2/ATLAS/WDs/"),
    "oreo": Path("/data/atlas/wds_subset/"),
    "node": Path("/orcd/data/kburdge/001/ATLAS/ATLAS_Lightcurves/"),
}


def _hostname():
    return socket.gethostname()


def atlas_base_dir(atlas_dir=None):
    """Resolve the root directory that holds the raw ATLAS light-curve files.

    Resolution order (first match wins):
    1) Explicit ``atlas_dir`` argument.
    2) Hostname-based defaults used by the atlas-quicklook project.
    """

    if atlas_dir is not None:
        base = Path(atlas_dir).expanduser()
    else:
        host = _hostname()
        base = None
        for key, path in _DEFAULT_ATLAS_HOST_PATHS.items():
            if host == key or host.startswith(key):
                base = path
                break
        if base is None:
            raise RuntimeError(
                "Could not determine ATLAS light-curve directory. "
                "Pass atlas_dir explicitly or add your hostname to _DEFAULT_ATLAS_HOST_PATHS."
            )

    if not base.exists():
        raise FileNotFoundError(
            f"ATLAS light-curve directory '{base}' does not exist. "
            "Pass atlas_dir explicitly or add your hostname to _DEFAULT_ATLAS_HOST_PATHS."
        )

    return base


def atlas_data_path(*parts, atlas_dir=None):
    """Join paths under the resolved ATLAS light-curve root."""

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
