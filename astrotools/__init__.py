"""astrotools package."""

from __future__ import annotations

import os
import subprocess
from typing import Iterable, List


def _find_modulecmd() -> str | None:
	"""Return the modulecmd path if available."""
	modulecmd = os.environ.get("MODULECMD")
	if modulecmd and os.path.isfile(modulecmd):
		return modulecmd
	for candidate in ("modulecmd", "modulecmd.tcl"):
		try:
			subprocess.run(
				[candidate, "--version"],
				check=False,
				stdout=subprocess.DEVNULL,
				stderr=subprocess.DEVNULL,
			)
			return candidate
		except FileNotFoundError:
			continue
	return None


def _load_modules(modules: Iterable[str]) -> None:
	modulecmd = _find_modulecmd()
	if modulecmd is None:
		return

	for module in modules:
		if not module:
			continue
		try:
			output = subprocess.check_output(
				[modulecmd, "python", "load", module],
				stderr=subprocess.STDOUT,
				text=True,
			)
		except subprocess.CalledProcessError:
			continue
		if output.strip():
			exec(output, {"os": os})


def _maybe_load_engaging_modules() -> None:
	if os.environ.get("ASTROTOOLS_DISABLE_MODULES"):
		return
	if os.environ.get("ASTROTOOLS_MODULES_LOADED"):
		return

	modules_env = os.environ.get("ASTROTOOLS_MODULES", "gcc/11.2.0,cuda/11.3")
	modules: List[str] = [entry.strip() for entry in modules_env.split(",")]
	_load_modules(modules)
	os.environ["ASTROTOOLS_MODULES_LOADED"] = "1"


_maybe_load_engaging_modules()

