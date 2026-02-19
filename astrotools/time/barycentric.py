"""Barycentric time conversions."""

from __future__ import annotations

from astropy.coordinates import BarycentricTrueEcliptic, EarthLocation, SkyCoord
from astropy import units as u
from astropy.units import UnrecognizedUnit
from astropy.time import Time


def _normalize_time_input(time, date_format):
    if not hasattr(time, "unit"):
        return time

    unit = time.unit
    if isinstance(unit, UnrecognizedUnit) and str(unit) == "seconds":
        time = time.value * u.s

    if date_format in {"mjd", "jd"}:
        try:
            return time.to(u.day)
        except Exception:
            return time

    return time


def bjd_time(
    time,
    ra,
    dec,
    date_format="mjd",
    telescope="Palomar",
    scale="tcb",
):
    """Return barycentric time as an astropy Time object."""

    coord = SkyCoord(ra, dec, unit="deg")
    time_obj = Time(_normalize_time_input(time, date_format), format=date_format, scale="utc")
    time_scaled = time_obj.tcb if scale == "tcb" else time_obj.tdb
    observatory = EarthLocation.of_site(telescope)
    correction = time_scaled.light_travel_time(
        coord, kind="barycentric", location=observatory
    )
    return time_scaled + correction


def bjd_convert(
    time,
    ra,
    dec,
    date_format="mjd",
    telescope="Palomar",
    scale="tcb",
):
    """Convert times to barycentric Julian dates."""

    return bjd_time(time, ra, dec, date_format, telescope, scale).mjd
