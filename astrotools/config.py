"""Machine-specific configuration for data paths.

This module provides hostname-based resolution of data directories across
different machines (oreo, engaging cluster, etc.).
"""

from __future__ import annotations

import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass
class MachineConfig:
    """Configuration for a specific machine."""
    
    name: str
    atlas_dir: Path | None = None
    tess_dir: Path | None = None
    catalog_dir: Path | None = None
    scratch_dir: Path | None = None
    wd_catalog: Path | None = None


# Machine-specific configurations
_MACHINE_CONFIGS: Dict[str, MachineConfig] = {
    "oreo": MachineConfig(
        name="oreo",
        atlas_dir=Path("/data/atlas/wds_subset/"),
        tess_dir=Path("/home/echickle/orcd/pool/TESS_Lightcurves"),
        catalog_dir=Path("/home/echickle/orcd/pool/gaia_edr3_wd_catalog"),
        scratch_dir=Path("/home/echickle/orcd/pool"),
        wd_catalog=Path("/home/echickle/orcd/pool/WDs.txt"),
    ),
    "node": MachineConfig(
        name="engaging",
        atlas_dir=Path("/orcd/data/kburdge/001/ATLAS/ATLAS_Lightcurves/"),
        tess_dir=Path("/home/echickle/orcd/scratch/TESS_Lightcurves"),
        catalog_dir=Path("/home/echickle/orcd/scratch/gaia_edr3_wd_catalog"),
        scratch_dir=Path("/home/echickle/orcd/scratch"),
        wd_catalog=Path("/home/echickle/orcd/scratch/WDs.txt"),
    ),
    "hypernova": MachineConfig(
        name="hypernova",
        atlas_dir=Path("/data2/ATLAS/WDs/"),
    ),
}


def get_hostname() -> str:
    """Get the current hostname."""
    return socket.gethostname()


def detect_machine() -> MachineConfig | None:
    """Detect the current machine and return its configuration.
    
    Returns
    -------
    MachineConfig | None
        Configuration for the current machine, or None if not recognized.
    """
    hostname = get_hostname()
    
    # Check for exact match first
    if hostname in _MACHINE_CONFIGS:
        return _MACHINE_CONFIGS[hostname]
    
    # Check for hostname prefix match
    for key, config in _MACHINE_CONFIGS.items():
        if hostname.startswith(key):
            return config
    
    return None


def get_atlas_dir(atlas_dir: str | Path | None = None) -> Path:
    """Resolve ATLAS light curve directory.
    
    Parameters
    ----------
    atlas_dir : str | Path | None
        Explicit directory path. If None, uses machine-specific default.
    
    Returns
    -------
    Path
        Resolved ATLAS directory path.
    
    Raises
    ------
    RuntimeError
        If directory cannot be determined from hostname.
    FileNotFoundError
        If the resolved directory does not exist.
    """
    if atlas_dir is not None:
        base = Path(atlas_dir).expanduser()
    else:
        config = detect_machine()
        if config is None or config.atlas_dir is None:
            raise RuntimeError(
                f"Could not determine ATLAS directory for hostname '{get_hostname()}'. "
                "Pass atlas_dir explicitly or add your machine to _MACHINE_CONFIGS."
            )
        base = config.atlas_dir
    
    if not base.exists():
        raise FileNotFoundError(
            f"ATLAS directory '{base}' does not exist. "
            "Verify the path or pass atlas_dir explicitly."
        )
    
    return base


def get_tess_dir(tess_dir: str | Path | None = None) -> Path:
    """Resolve TESS light curve directory.
    
    Parameters
    ----------
    tess_dir : str | Path | None
        Explicit directory path. If None, uses machine-specific default.
    
    Returns
    -------
    Path
        Resolved TESS directory path.
    
    Raises
    ------
    RuntimeError
        If directory cannot be determined from hostname.
    FileNotFoundError
        If the resolved directory does not exist.
    """
    if tess_dir is not None:
        base = Path(tess_dir).expanduser()
    else:
        config = detect_machine()
        if config is None or config.tess_dir is None:
            raise RuntimeError(
                f"Could not determine TESS directory for hostname '{get_hostname()}'. "
                "Pass tess_dir explicitly or add your machine to _MACHINE_CONFIGS."
            )
        base = config.tess_dir
    
    if not base.exists():
        raise FileNotFoundError(
            f"TESS directory '{base}' does not exist. "
            "Verify the path or pass tess_dir explicitly."
        )
    
    return base


def get_catalog_dir(catalog_dir: str | Path | None = None) -> Path:
    """Resolve catalog directory.
    
    Parameters
    ----------
    catalog_dir : str | Path | None
        Explicit directory path. If None, uses machine-specific default.
    
    Returns
    -------
    Path
        Resolved catalog directory path.
    
    Raises
    ------
    RuntimeError
        If directory cannot be determined from hostname.
    FileNotFoundError
        If the resolved directory does not exist.
    """
    if catalog_dir is not None:
        base = Path(catalog_dir).expanduser()
    else:
        config = detect_machine()
        if config is None or config.catalog_dir is None:
            raise RuntimeError(
                f"Could not determine catalog directory for hostname '{get_hostname()}'. "
                "Pass catalog_dir explicitly or add your machine to _MACHINE_CONFIGS."
            )
        base = config.catalog_dir
    
    if not base.exists():
        raise FileNotFoundError(
            f"Catalog directory '{base}' does not exist. "
            "Verify the path or pass catalog_dir explicitly."
        )
    
    return base


def get_wd_catalog(catalog_path: str | Path | None = None) -> Path:
    """Resolve white dwarf catalog file path.
    
    Parameters
    ----------
    catalog_path : str | Path | None
        Explicit catalog file path. If None, uses machine-specific default.
    
    Returns
    -------
    Path
        Resolved catalog file path.
    
    Raises
    ------
    RuntimeError
        If catalog cannot be determined from hostname.
    FileNotFoundError
        If the resolved catalog file does not exist.
    """
    if catalog_path is not None:
        path = Path(catalog_path).expanduser()
    else:
        config = detect_machine()
        if config is None or config.wd_catalog is None:
            raise RuntimeError(
                f"Could not determine WD catalog for hostname '{get_hostname()}'. "
                "Pass catalog_path explicitly or add your machine to _MACHINE_CONFIGS."
            )
        path = config.wd_catalog
    
    if not path.exists():
        raise FileNotFoundError(
            f"WD catalog '{path}' does not exist. "
            "Verify the path or pass catalog_path explicitly."
        )
    
    return path


def get_scratch_dir(scratch_dir: str | Path | None = None) -> Path:
    """Resolve scratch directory for temporary/output files.
    
    Parameters
    ----------
    scratch_dir : str | Path | None
        Explicit directory path. If None, uses machine-specific default.
    
    Returns
    -------
    Path
        Resolved scratch directory path.
    
    Raises
    ------
    RuntimeError
        If directory cannot be determined from hostname.
    """
    if scratch_dir is not None:
        base = Path(scratch_dir).expanduser()
    else:
        config = detect_machine()
        if config is None or config.scratch_dir is None:
            raise RuntimeError(
                f"Could not determine scratch directory for hostname '{get_hostname()}'. "
                "Pass scratch_dir explicitly or add your machine to _MACHINE_CONFIGS."
            )
        base = config.scratch_dir
    
    # Create scratch directory if it doesn't exist
    base.mkdir(parents=True, exist_ok=True)
    
    return base


def print_config(verbose: bool = False) -> None:
    """Print the current machine configuration.
    
    Parameters
    ----------
    verbose : bool
        If True, print all machine configurations.
    """
    hostname = get_hostname()
    config = detect_machine()
    
    print(f"Hostname: {hostname}")
    
    if config is None:
        print("Machine: UNKNOWN (not configured)")
        return
    
    print(f"Machine: {config.name}")
    print("\nConfigured paths:")
    
    if config.atlas_dir:
        exists = "✓" if config.atlas_dir.exists() else "✗"
        print(f"  ATLAS:   {exists} {config.atlas_dir}")
    
    if config.tess_dir:
        exists = "✓" if config.tess_dir.exists() else "✗"
        print(f"  TESS:    {exists} {config.tess_dir}")
    
    if config.catalog_dir:
        exists = "✓" if config.catalog_dir.exists() else "✗"
        print(f"  Catalog: {exists} {config.catalog_dir}")
    
    if config.wd_catalog:
        exists = "✓" if config.wd_catalog.exists() else "✗"
        print(f"  WD Cat:  {exists} {config.wd_catalog}")
    
    if config.scratch_dir:
        exists = "✓" if config.scratch_dir.exists() else "✗"
        print(f"  Scratch: {exists} {config.scratch_dir}")
    
    if verbose:
        print("\n" + "=" * 60)
        print("All configured machines:")
        print("=" * 60)
        for key, cfg in _MACHINE_CONFIGS.items():
            print(f"\n{key} ({cfg.name}):")
            if cfg.atlas_dir:
                print(f"  ATLAS:   {cfg.atlas_dir}")
            if cfg.tess_dir:
                print(f"  TESS:    {cfg.tess_dir}")
            if cfg.catalog_dir:
                print(f"  Catalog: {cfg.catalog_dir}")
            if cfg.wd_catalog:
                print(f"  WD Cat:  {cfg.wd_catalog}")
            if cfg.scratch_dir:
                print(f"  Scratch: {cfg.scratch_dir}")


if __name__ == "__main__":
    print_config(verbose=True)
