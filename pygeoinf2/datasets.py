"""The shipped data tables, and the explicit downloads that refresh them.

Two small catalogues ship with the package, a Global Seismograph Network
station list and a USGS earthquake catalogue, so the worked examples have a
real acquisition geometry. The sphere's :meth:`~pygeoinf2.symmetric_space.sphere.Sphere.stations`
and :meth:`~pygeoinf2.symmetric_space.sphere.Sphere.earthquakes` read them
through :func:`read_table`, which takes a copy in the user's cache directory
before the bundled one, so a refreshed catalogue takes effect without any
other change.

The refresh is explicit. :func:`download_gsn_stations` and
:func:`download_usgs_earthquakes` fetch from the IRIS and USGS web services
and write into :func:`cache_directory`, never into the package, which may
sit in a read-only tree. Nothing else here touches the network, and nothing
fetches on a caller's behalf: asking for more events than the table holds
is refused with the download named, rather than answered with a fetch the
caller did not ask for (DESIGN §21.2, §82). v1's ``datasets`` and
``config`` modules, with the automatic fetch left out.

The cache directory is ``PYGEOINF_CACHE_DIR`` when set, otherwise
``$XDG_CACHE_HOME/pygeoinf`` or ``~/.cache/pygeoinf`` on Unix and
``%LOCALAPPDATA%\\pygeoinf\\Cache`` on Windows, resolved at each call so a
test can point it elsewhere.
"""

from __future__ import annotations

import csv
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Sequence

import numpy as np

__all__ = [
    "GSN_STATIONS",
    "USGS_EVENTS",
    "cache_directory",
    "table_path",
    "read_table",
    "download_gsn_stations",
    "download_usgs_earthquakes",
]

GSN_STATIONS = "gsn_stations.csv"
"""The station table: ``Station, Latitude, Longitude``."""

USGS_EVENTS = "usgs_event_cache.csv"
"""The earthquake table, in the USGS query's CSV format."""

_IRIS_STATIONS = "https://service.iris.edu/fdsnws/station/1/query"
_USGS_EVENTS = "https://earthquake.usgs.gov/fdsnws/event/1/query"


def cache_directory() -> Path:
    """The writable directory downloads go to.

    Returns:
        ``PYGEOINF_CACHE_DIR`` if set; otherwise the platform's user cache
        location with ``pygeoinf`` appended. Not created here.
    """
    override = os.environ.get("PYGEOINF_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(
            Path("~/AppData/Local").expanduser()
        )
        return Path(base) / "pygeoinf" / "Cache"
    base = os.environ.get("XDG_CACHE_HOME") or str(Path("~/.cache").expanduser())
    return Path(base) / "pygeoinf"


def table_path(name: str, /) -> Path | None:
    """Where a table is read from: the cached copy, else the bundled one.

    Args:
        name: the file name, such as :data:`GSN_STATIONS`.

    Returns:
        The path, or ``None`` if neither copy exists.
    """
    from importlib.resources import files

    cached = cache_directory() / name
    if cached.exists():
        return cached
    bundled = files("pygeoinf2.data") / name
    if bundled.is_file():
        return Path(str(bundled))
    return None


def read_table(name: str, /) -> dict[str, np.ndarray]:
    """Read a table into arrays keyed by column, numeric where every entry parses.

    Args:
        name: the file name.

    Returns:
        One array per column: floats where the column is numeric, objects
        otherwise.

    Raises:
        FileNotFoundError: if neither a cached nor a bundled copy exists.
        ValueError: if the table has no rows.
    """
    path = table_path(name)
    if path is None:
        raise FileNotFoundError(
            f"No table {name!r} is cached or bundled; download_gsn_stations() and "
            "download_usgs_earthquakes() write one."
        )
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    if not rows:
        raise ValueError(f"{name} is empty.")
    table: dict[str, np.ndarray] = {}
    for column in rows[0]:
        values = [row[column] for row in rows]
        try:
            table[column] = np.array([float(value) for value in values])
        except ValueError:
            table[column] = np.array(values, dtype=object)
    return table


def _fetch(url: str, params: dict[str, Any], /, *, timeout: float) -> str:
    """One GET, decoded. The only place the network is touched, so a test
    replaces this and nothing else."""
    full = f"{url}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(full, timeout=timeout) as response:
        return response.read().decode("utf-8")


def _cache_file(name: str) -> Path:
    directory = cache_directory()
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name


def download_gsn_stations(*, force: bool = False, timeout: float = 10.0) -> Path:
    """Fetch the Global Seismograph Network station list from IRIS into the cache.

    Networks ``IU`` and ``II``, written as ``Station, Latitude, Longitude``
    under :data:`GSN_STATIONS`, which is then what
    :meth:`~pygeoinf2.symmetric_space.sphere.Sphere.stations` reads.

    Args:
        force: fetch again even if a cached copy exists.
        timeout: seconds to wait for the service.

    Returns:
        The path written, or the existing cached copy.

    Raises:
        RuntimeError: if the fetch fails or returns nothing usable.
    """
    target = _cache_file(GSN_STATIONS)
    if target.exists() and not force:
        return target
    params = {"network": "IU,II", "level": "station", "format": "text"}
    try:
        text = _fetch(_IRIS_STATIONS, params, timeout=timeout)
    except Exception as error:
        raise RuntimeError(
            f"Fetching the GSN stations from IRIS failed: {error}"
        ) from error
    stations: list[Sequence[Any]] = []
    for line in text.strip().splitlines()[1:]:
        parts = line.split("|")
        if len(parts) >= 4:
            stations.append([parts[1].strip(), float(parts[2]), float(parts[3])])
    if not stations:
        raise RuntimeError("IRIS returned no stations.")
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Station", "Latitude", "Longitude"])
        writer.writerows(stations)
    return target


def download_usgs_earthquakes(
    *,
    minimum_magnitude: float | None = 5.0,
    start_time: str | None = None,
    end_time: str | None = None,
    minimum_depth: float | None = None,
    maximum_depth: float | None = None,
    bounding_box: Sequence[float] | None = None,
    limit: int = 2000,
    force: bool = False,
    filename: str = USGS_EVENTS,
    timeout: float = 20.0,
) -> Path:
    """Fetch a filtered earthquake catalogue from the USGS into the cache.

    Written in the service's own CSV format. Under the default *filename* it
    is what :meth:`~pygeoinf2.symmetric_space.sphere.Sphere.earthquakes`
    reads from then on; another name keeps the seed table in use.

    Args:
        minimum_magnitude: the smallest magnitude to include.
        start_time: the earliest event, as an ISO date or datetime.
        end_time: the latest.
        minimum_depth: the shallowest, in kilometres.
        maximum_depth: the deepest.
        bounding_box: ``(min_latitude, max_latitude, min_longitude,
            max_longitude)`` in degrees.
        limit: at most this many events, the most recent first.
        force: fetch again even if the file exists.
        filename: the name to write under in the cache.
        timeout: seconds to wait for the service.

    Returns:
        The path written, or the existing cached copy.

    Raises:
        ValueError: for a non-positive limit or a bounding box of the wrong
            length.
        RuntimeError: if the fetch fails or returns no events.
    """
    if limit < 1:
        raise ValueError(f"The limit must be positive, got {limit}.")
    if bounding_box is not None and len(bounding_box) != 4:
        raise ValueError(
            "A bounding box is (min_latitude, max_latitude, min_longitude, max_longitude)."
        )
    target = _cache_file(filename)
    if target.exists() and not force:
        return target
    params: dict[str, Any] = {"format": "csv", "limit": int(limit), "orderby": "time"}
    if minimum_magnitude is not None:
        params["minmagnitude"] = minimum_magnitude
    if start_time is not None:
        params["starttime"] = start_time
    if end_time is not None:
        params["endtime"] = end_time
    if minimum_depth is not None:
        params["mindepth"] = minimum_depth
    if maximum_depth is not None:
        params["maxdepth"] = maximum_depth
    if bounding_box is not None:
        params["minlatitude"], params["maxlatitude"] = bounding_box[0], bounding_box[1]
        params["minlongitude"], params["maxlongitude"] = (
            bounding_box[2],
            bounding_box[3],
        )
    try:
        text = _fetch(_USGS_EVENTS, params, timeout=timeout)
    except Exception as error:
        raise RuntimeError(
            f"Fetching the earthquakes from the USGS failed: {error}"
        ) from error
    if len(text.strip().splitlines()) < 2:
        raise RuntimeError("The USGS returned no events for these filters.")
    target.write_text(text, encoding="utf-8")
    return target
