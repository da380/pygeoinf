"""
pygeoinf/datasets.py

Provides access to built-in datasets for testing, benchmarking,
and visualization across the pygeoinf package.
"""

import os
import csv
import random
import urllib.request
import urllib.parse
from typing import List, Tuple, Union

# Import the centralized path
from .config import DATADIR

# Define the specific file path
_CSV_PATH = os.path.join(DATADIR, "gsn_stations.csv")


def download_gsn_stations(force: bool = False) -> None:
    """
    Fetches the Global Seismograph Network (GSN) stations from the IRIS
    FDSN API and saves them to a local CSV file in the data/ directory.
    """
    if os.path.exists(_CSV_PATH) and not force:
        return

    print("pygeoinf: Local dataset missing. Fetching station data from IRIS...")

    # Ensure the central DATADIR exists before writing!
    os.makedirs(DATADIR, exist_ok=True)

    url = "http://service.iris.edu/fdsnws/station/1/query"
    params = {"network": "IU,II", "level": "station", "format": "text"}

    full_url = f"{url}?{urllib.parse.urlencode(params)}"
    stations = []

    try:
        with urllib.request.urlopen(full_url, timeout=10) as response:
            lines = response.read().decode("utf-8").strip().split("\n")

            for line in lines[1:]:
                parts = line.split("|")
                if len(parts) >= 4:
                    stations.append([parts[1], float(parts[2]), float(parts[3])])

        with open(_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Station", "Latitude", "Longitude"])
            writer.writerows(stations)

        print(f"pygeoinf: Successfully saved {len(stations)} stations to {_CSV_PATH}")

    except Exception as e:
        raise RuntimeError(f"Failed to download GSN stations from IRIS. Error: {e}")


def load_gsn_stations(
    n_stations: int = None, include_names: bool = False
) -> Union[List[Tuple[float, float]], List[Tuple[str, float, float]]]:
    """
    Loads a representative global set of seismic stations from the GSN.

    If the internal CSV file is missing, this function will attempt to
    automatically download it from IRIS into the pygeoinf/data/ directory.

    Args:
        n_stations: If provided, returns a random subsample of this size.
                    If greater than the total available stations, returns all.
        include_names: If True, returns (Name, Latitude, Longitude).
                       If False, returns pure (Latitude, Longitude) tuples.

    Returns:
        A list of station tuples in degrees.
    """
    _CSV_PATH = os.path.join(DATADIR, "gsn_stations.csv")

    if not os.path.exists(_CSV_PATH):
        download_gsn_stations()

    stations = []
    with open(_CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lat = float(row["Latitude"])
            lon = float(row["Longitude"])
            if include_names:
                stations.append((row["Station"], lat, lon))
            else:
                stations.append((lat, lon))

    # Sub-sample if requested and mathematically valid
    if n_stations is not None and n_stations < len(stations):
        return random.sample(stations, n_stations)

    return stations


def download_usgs_earthquakes(
    min_magnitude: float = 5.0,
    start_time: str = None,
    end_time: str = None,
    min_depth: float = None,
    max_depth: float = None,
    bbox: Tuple[float, float, float, float] = None,
    limit: int = 2000,
    force: bool = False,
    filename: str = "usgs_events_filtered.csv",
) -> None:
    """
    Fetches a filtered catalog of earthquakes from the USGS API and saves it
    to a CSV in the centralized DATADIR.
    """
    csv_path = os.path.join(DATADIR, filename)

    if os.path.exists(csv_path) and not force:
        return

    print(f"pygeoinf: Fetching up to {limit} earthquakes from USGS...")
    os.makedirs(DATADIR, exist_ok=True)

    params = {"format": "csv", "limit": limit, "orderby": "time"}

    if min_magnitude is not None:
        params["minmagnitude"] = min_magnitude
    if start_time is not None:
        params["starttime"] = start_time
    if end_time is not None:
        params["endtime"] = end_time
    if min_depth is not None:
        params["mindepth"] = min_depth
    if max_depth is not None:
        params["maxdepth"] = max_depth
    if bbox is not None:
        params["minlatitude"] = bbox[0]
        params["maxlatitude"] = bbox[1]
        params["minlongitude"] = bbox[2]
        params["maxlongitude"] = bbox[3]

    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    full_url = f"{url}?{urllib.parse.urlencode(params)}"

    try:
        with urllib.request.urlopen(full_url, timeout=20) as response:
            data = response.read().decode("utf-8")

        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(data)

        num_events = len(data.strip().split("\n")) - 1
        print(f"pygeoinf: Successfully saved {num_events} events to {csv_path}")

    except Exception as e:
        raise RuntimeError(f"Failed to download USGS events. Error: {e}")


def sample_earthquakes(
    n_events: int, min_magnitude: float = 5.0
) -> List[Tuple[float, float, float]]:
    """
    Returns a random subsample of real earthquake locations.

    If the local cache does not contain enough events to satisfy the request,
    it automatically fetches a larger catalog from the USGS to rebuild the cache.

    Args:
        n_events: The exact number of earthquake locations to return.
        min_magnitude: The minimum magnitude to use if a new download is required.

    Returns:
        A list of tuples: (Latitude, Longitude, Depth_in_km).
    """
    cache_filename = "usgs_event_cache.csv"
    cache_path = os.path.join(DATADIR, cache_filename)

    events = []

    # 1. Try loading from the existing cache
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                events.append(
                    (
                        float(row["latitude"]),
                        float(row["longitude"]),
                        float(row["depth"]),
                    )
                )

    # 2. Check if the cache is large enough
    if len(events) >= n_events:
        # Use random.sample to grab unique items without replacement
        return random.sample(events, n_events)

    # 3. Cache is too small (or doesn't exist). Download a new one!
    # Smart fetching: Always download at least 2000, or the requested amount + a 20% buffer.
    # This prevents hitting the FDSN API repeatedly if the user slowly increases n_events.
    fetch_limit = max(2000, int(n_events * 1.2))
    print(
        f"pygeoinf: Local cache only has {len(events)} events. Fetching {fetch_limit} to build a robust cache..."
    )

    download_usgs_earthquakes(
        min_magnitude=min_magnitude,
        limit=fetch_limit,
        filename=cache_filename,
        force=True,  # Overwrite the old, insufficient cache
    )

    # 4. Reload the newly downloaded cache
    events = []
    with open(cache_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            events.append(
                (float(row["latitude"]), float(row["longitude"]), float(row["depth"]))
            )

    # 5. Return the exact sample size requested
    # If the API returned fewer events than requested (e.g., asked for 100,000 Mag 9.0s),
    # we just return whatever we actually managed to get.
    return random.sample(events, min(n_events, len(events)))
