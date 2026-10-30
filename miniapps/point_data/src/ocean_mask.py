"""The ocean mask, from the Natural Earth land polygons.

Cartopy fetches the 110m land shapefile into its cache on first use;
shapely answers the point-in-polygon questions, vectorized. The sphere's
own ``random_domain_points`` does the same test one point at a time; here
tens of thousands of points are tested at once.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import shapely
from cartopy.io import shapereader


def land_polygons() -> Any:
    """The Natural Earth land, as one shapely geometry."""
    path = shapereader.natural_earth(
        resolution="110m", category="physical", name="land"
    )
    return shapely.union_all(list(shapereader.Reader(path).geometries()))


def on_land(land: Any, latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    """Whether each ``(latitude, longitude)`` in degrees lies on land."""
    wrapped = (np.asarray(longitudes, dtype=float) + 180.0) % 360.0 - 180.0
    return shapely.contains_xy(land, wrapped, np.asarray(latitudes, dtype=float))


def ocean_points(count: int, land: Any, /, *, rng: np.random.Generator) -> np.ndarray:
    """``count`` points uniform over the sphere's area, kept where not on land.

    Returns:
        A ``(count, 2)`` array of ``(latitude, longitude)`` in degrees.
    """
    kept: list[np.ndarray] = []
    total = 0
    while total < count:
        draw = max(100, int(1.6 * (count - total)))
        latitudes = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, draw)))
        longitudes = rng.uniform(-180.0, 180.0, draw)
        sea = ~on_land(land, latitudes, longitudes)
        kept.append(np.column_stack([latitudes, longitudes])[sea])
        total += int(sea.sum())
    return np.concatenate(kept)[:count]


def grid_degrees(space: Any) -> tuple[np.ndarray, np.ndarray]:
    """``(longitude, latitude)`` in degrees at every grid cell of a sphere."""
    latitudes = 90.0 - np.degrees(space.colatitudes)
    return np.meshgrid(np.degrees(space.longitudes), latitudes)


def land_mask(space: Any, land: Any) -> np.ndarray:
    """The grid cells of a sphere that lie on land."""
    longitude, latitude = grid_degrees(space)
    return on_land(land, latitude.ravel(), longitude.ravel()).reshape(latitude.shape)


def area_weights(space: Any) -> np.ndarray:
    """``cos(latitude)`` per grid cell, for area-weighted averages of a map."""
    latitudes = 90.0 - np.degrees(space.colatitudes)
    weights = np.cos(np.radians(latitudes))
    return np.repeat(weights[:, None], space.longitudes.size, axis=1)


def geodesic_bump(space: Any, center: np.ndarray, width: float) -> Any:
    """A Gaussian bump of unit peak about a center, as a field of the sphere."""
    longitude, latitude = grid_degrees(space)
    lat0, lon0 = np.radians(center)
    lat, lon = np.radians(latitude), np.radians(longitude)
    cosine = np.sin(lat0) * np.sin(lat) + np.cos(lat0) * np.cos(lat) * np.cos(
        lon - lon0
    )
    distance = np.arccos(np.clip(cosine, -1.0, 1.0))
    return space.from_grid_values(np.exp(-0.5 * (distance / width) ** 2))
