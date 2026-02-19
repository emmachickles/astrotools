"""Path helpers for ATLAS data products."""

from pathlib import Path
import socket

from .config import atlas_base_dir

_DEFAULT_BLS_HOST_PATHS = {
    "hypernova": Path("/data/Bulk_BLS_ATLAS/"),
    "node": Path("/orcd/data/kburdge/001/ATLAS/ATLAS_BLS/"),
}


def _hostname() -> str:
    return socket.gethostname()


def data_path(*parts, atlas_dir=None):
    """Join paths under the ATLAS light-curve root."""

    return atlas_base_dir(atlas_dir).joinpath(*map(Path, parts))


def bls_path(*parts, bls_dir=None):
    """Join paths under the ATLAS BLS results root."""

    if bls_dir is not None:
        base = Path(bls_dir).expanduser()
    else:
        host = _hostname()
        base = None
        for key, path in _DEFAULT_BLS_HOST_PATHS.items():
            if host == key or host.startswith(key):
                base = path
                break
        if base is None:
            raise RuntimeError(
                "Could not determine ATLAS BLS directory. "
                "Pass bls_dir explicitly or add your hostname to _DEFAULT_BLS_HOST_PATHS."
            )

    if not base.exists():
        raise FileNotFoundError(
            f"ATLAS BLS directory '{base}' does not exist. "
            "Pass bls_dir explicitly or add your hostname to _DEFAULT_BLS_HOST_PATHS."
        )

    return base.joinpath(*map(Path, parts))


def lc_path(source_id, atlas_dir=None):
    """Return the light-curve file path for a source id."""

    return data_path(str(source_id), atlas_dir=atlas_dir)
