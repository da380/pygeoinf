"""
Depth Coordinate System (paper_demos copy)

Utilities for converting depths/radii/normalized coordinates.
"""

import numpy as np
from typing import Union

EARTH_RADIUS_KM = 6371.0


class DepthCoordinateSystem:
    EARTH_RADIUS_KM = EARTH_RADIUS_KM

    @staticmethod
    def depth_to_normalized(depth_km: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return depth_km / DepthCoordinateSystem.EARTH_RADIUS_KM

    @staticmethod
    def normalized_to_depth(normalized: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return normalized * DepthCoordinateSystem.EARTH_RADIUS_KM

    @staticmethod
    def radius_to_depth(radius_km: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return DepthCoordinateSystem.EARTH_RADIUS_KM - radius_km

    @staticmethod
    def depth_to_radius(depth_km: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return DepthCoordinateSystem.EARTH_RADIUS_KM - depth_km

    @staticmethod
    def radius_to_normalized(radius_km: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        depth_km = DepthCoordinateSystem.radius_to_depth(radius_km)
        return DepthCoordinateSystem.depth_to_normalized(depth_km)

    @staticmethod
    def normalized_to_radius(normalized: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        depth_km = DepthCoordinateSystem.normalized_to_depth(normalized)
        return DepthCoordinateSystem.depth_to_radius(depth_km)

    @staticmethod
    def validate_depth(depth_km: Union[float, np.ndarray], allow_negative: bool = False) -> bool:
        depth_array = np.atleast_1d(depth_km)
        if allow_negative:
            return np.all(depth_array <= DepthCoordinateSystem.EARTH_RADIUS_KM)
        else:
            return np.all((depth_array >= 0) & (depth_array <= DepthCoordinateSystem.EARTH_RADIUS_KM))

    @staticmethod
    def validate_radius(radius_km: Union[float, np.ndarray]) -> bool:
        radius_array = np.atleast_1d(radius_km)
        return np.all((radius_array >= 0) & (radius_array <= DepthCoordinateSystem.EARTH_RADIUS_KM))

    @staticmethod
    def validate_normalized(normalized: Union[float, np.ndarray], tolerance: float = 1e-10) -> bool:
        norm_array = np.atleast_1d(normalized)
        return np.all((norm_array >= -tolerance) & (norm_array <= 1.0 + tolerance))

    @staticmethod
    def get_major_discontinuities() -> dict:
        discontinuities = {
            'Surface': (0.0, 6371.0),
            'Moho': (24.4, 6346.6),
            '410': (410.0, 5961.0),
            '660': (660.0, 5711.0),
            'CMB': (2891.0, 3480.0),
            'ICB': (5153.5, 1217.5),
            'Center': (6371.0, 0.0),
        }
        return discontinuities

    @staticmethod
    def get_major_layers() -> dict:
        layers = {
            'Crust': (0.0, 24.4),
            'Upper_Mantle': (24.4, 660.0),
            'Transition_Zone': (410.0, 660.0),
            'Lower_Mantle': (660.0, 2891.0),
            'Mantle': (24.4, 2891.0),
            'Outer_Core': (2891.0, 5153.5),
            'Inner_Core': (5153.5, 6371.0),
            'Core': (2891.0, 6371.0),
        }
        return layers

    @classmethod
    def info(cls) -> str:
        info_str = f"""Depth Coordinate System\n========================\nEarth Radius: {cls.EARTH_RADIUS_KM} km\n\nCoordinate Systems:\n  - depth_km: Depth from surface [0, {cls.EARTH_RADIUS_KM}] km\n  - radius_km: Radius from center [0, {cls.EARTH_RADIUS_KM}] km\n  - normalized: Normalized depth [0, 1]\n\nNormalization Convention:\n  0.0 → Surface (0 km depth, {cls.EARTH_RADIUS_KM} km radius)\n  1.0 → Center ({cls.EARTH_RADIUS_KM} km depth, 0 km radius)\n\nMajor Discontinuities:"""
        disc = cls.get_major_discontinuities()
        for name, (depth, radius) in disc.items():
            norm = depth / cls.EARTH_RADIUS_KM
            info_str += f"\n  {name:20s}: {depth:7.1f} km depth, {radius:7.1f} km radius, {norm:6.4f} normalized"
        return info_str
