"""
Discontinuity Kernel Handling

Classes:
    DiscontinuityKernel: Handle topography sensitivity at discontinuities
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from .depth_coordinates import DepthCoordinateSystem


class DiscontinuityKernel:
    def __init__(self, depths_km: np.ndarray, values: np.ndarray):
        if len(depths_km) != len(values):
            raise ValueError("depths_km and values must have same length")

        if len(depths_km) > 0 and not DepthCoordinateSystem.validate_depth(depths_km):
            raise ValueError("depths_km must be in valid Earth range")

        self.depths_km = np.asarray(depths_km)
        self.values = np.asarray(values)
        self.n_discontinuities = len(depths_km)
        self.depths_normalized = DepthCoordinateSystem.depth_to_normalized(depths_km)

        if self.n_discontinuities > 0:
            sort_idx = np.argsort(self.depths_km)
            self.depths_km = self.depths_km[sort_idx]
            self.values = self.values[sort_idx]
            self.depths_normalized = self.depths_normalized[sort_idx]

    def to_euclidean_vector(self) -> np.ndarray:
        return self.values.copy()

    def get_discontinuity_map(self, use_normalized: bool = False) -> Dict[float, float]:
        if use_normalized:
            return dict(zip(self.depths_normalized, self.values))
        else:
            return dict(zip(self.depths_km, self.values))

    def get_value_at_depth(self, depth_km: float, tolerance: float = 1.0) -> Optional[float]:
        if self.n_discontinuities == 0:
            return None

        distances = np.abs(self.depths_km - depth_km)
        min_idx = np.argmin(distances)

        if distances[min_idx] <= tolerance:
            return self.values[min_idx]
        else:
            return None

    def identify_discontinuities(self, known_discontinuities: Optional[Dict[str, float]] = None) -> Dict[str, Tuple[float, float]]:
        if known_discontinuities is None:
            disc_data = DepthCoordinateSystem.get_major_discontinuities()
            known_discontinuities = {name: depth for name, (depth, _) in disc_data.items()}

        identified = {}
        tolerance = 50.0

        for name, expected_depth in known_discontinuities.items():
            value = self.get_value_at_depth(expected_depth, tolerance=tolerance)
            if value is not None:
                distances = np.abs(self.depths_km - expected_depth)
                min_idx = np.argmin(distances)
                actual_depth = self.depths_km[min_idx]
                identified[name] = (actual_depth, value)

        return identified

    def to_point_evaluation_operators(self, lebesgue_space) -> List:
        raise NotImplementedError("Point evaluation operators require pygeoinf integration.")

    def plot(self, ax=None, use_depth: bool = True, **kwargs):
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        if self.n_discontinuities == 0:
            ax.text(0.5, 0.5, 'No discontinuities', ha='center', va='center', transform=ax.transAxes)
            return ax

        if use_depth:
            x = self.depths_km
            xlabel = 'Depth [km]'
        else:
            x = self.depths_normalized
            xlabel = 'Normalized Depth'

        markerline, stemlines, baseline = ax.stem(x, self.values,
                                                  linefmt=kwargs.get('linefmt', 'C0-'),
                                                  markerfmt=kwargs.get('markerfmt', 'C0o'),
                                                  basefmt=kwargs.get('basefmt', 'k-'))

        if use_depth:
            identified = self.identify_discontinuities()
            labeled = set()

            for i, (depth, value) in enumerate(zip(self.depths_km, self.values)):
                label = None
                for name, (id_depth, id_value) in identified.items():
                    if abs(depth - id_depth) < 1.0 and name not in labeled:
                        label = name
                        labeled.add(name)
                        break

                if label is None:
                    label = f"{depth:.0f} km"

                ax.text(depth, value, f"  {label}", ha='left', va='bottom', fontsize=8)

        ax.set_xlabel(xlabel)
        ax.set_ylabel('Topography Sensitivity')
        ax.set_title('Discontinuity Kernel')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        return ax

    def __repr__(self) -> str:
        if self.n_discontinuities == 0:
            return "DiscontinuityKernel(n=0, empty)"

        depth_range = f"[{self.depths_km.min():.1f}, {self.depths_km.max():.1f}]"
        value_range = f"[{self.values.min():.2e}, {self.values.max():.2e}]"
        return (
            f"DiscontinuityKernel(n={self.n_discontinuities}, depths={depth_range} km, values={value_range})"
        )

    def __len__(self) -> int:
        return self.n_discontinuities

    def __getitem__(self, index) -> Tuple[float, float]:
        return self.depths_km[index], self.values[index]
