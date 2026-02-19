astrotools
==========

astrotools is a personal toolbox of astronomy analysis code from my research. It’s a mix of reusable helpers, small scripts, and notes for time-series data, spectra, SED analysis, and more.

Spectral utilities
------------------
- Modules live under astrotools.spectral
- Model grids default to /data/models/ELM (override with ASTROTOOLS_MODEL_DIR)
- Atmosphere classes supported by `load_atmosphere_grid`: ELM/DA (H-rich) and DB (He-rich)

Migration notes (from scattered scripts)
----------------------------------------
- Legacy SPECTRAL_UTILS names are available as compatibility aliases in astrotools.spectral:
	- load_models, interpolate, interpolate_linear
	- broadening, convert_flux_density, convert_to_physical
	- shift_wavelength, clip_outliers, linear_model, line_model, extract_lines
- Binary/SED helpers extracted from Joint.py are available in astrotools.spectral:
	- roche_lobe_radius (alias: calculate_radius)
	- secondary_rv_semiamplitude (alias: calculate_rv)
	- log_surface_gravity (alias: calculate_logg)
	- gaussian_log_probability (alias: gauss_prob)

User Install
------------
- with uv (recommended)
```
uv sync
```

- with pip
```
python -m venv .venv
source .venv/bin/activate
pip install -e .
```


