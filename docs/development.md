Development Notes
=================

uv Workflow
-----------

This repo uses `uv` for dependency management and reproducible environments.


- Add dependencies

```
uv add astropy
```

- Sync environment

```
uv sync
```

- Run code in the python environment
```
uv run python script.py
```