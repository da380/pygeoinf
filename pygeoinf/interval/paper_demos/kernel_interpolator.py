"""
Kernel Interpolation

Interpolation utilities for converting discrete kernel arrays into
continuous interpolators and function-space coefficients.
"""

import numpy as np
from typing import Optional, Callable, Union, Tuple
from pygeoinf.interval.functions import Function
from scipy.interpolate import interp1d, UnivariateSpline
from .depth_coordinates import DepthCoordinateSystem


class KernelInterpolator:
    def __init__(self,
                 depths_km: np.ndarray,
                 values: np.ndarray,
                 method: str = 'cubic',
                 smoothing: Optional[float] = None,
                 extrapolate: str = 'zero'):
        if len(depths_km) != len(values):
            raise ValueError("depths_km and values must have same length")

        if len(depths_km) < 2:
            raise ValueError("Need at least 2 data points for interpolation")

        if not DepthCoordinateSystem.validate_depth(depths_km):
            raise ValueError("depths_km must be in valid Earth range")

        self.depths_km = np.asarray(depths_km)
        self.values = np.asarray(values)
        self.method = method
        self.smoothing = smoothing
        self.extrapolate = extrapolate

        self.depths_normalized = DepthCoordinateSystem.depth_to_normalized(depths_km)

        if not np.all(np.diff(self.depths_normalized) >= 0):
            sort_idx = np.argsort(self.depths_normalized)
            self.depths_normalized = self.depths_normalized[sort_idx]
            self.values = self.values[sort_idx]

        self._interpolator = self._create_interpolator()

    def _create_interpolator(self) -> Callable:
        if self.extrapolate == 'zero':
            fill_value = 0.0
        elif self.extrapolate == 'constant':
            fill_value = (self.values[0], self.values[-1])
        elif self.extrapolate == 'error':
            fill_value = None
        else:
            raise ValueError(f"Unknown extrapolate method: {self.extrapolate}")

        if self.method == 'linear':
            return interp1d(self.depths_normalized, self.values, kind='linear',
                            fill_value=fill_value,
                            bounds_error=(self.extrapolate == 'error'),
                            assume_sorted=True)

        elif self.method == 'cubic':
            if len(self.depths_normalized) < 4:
                return interp1d(self.depths_normalized, self.values, kind='linear',
                                fill_value=fill_value,
                                bounds_error=(self.extrapolate == 'error'),
                                assume_sorted=True)

            return interp1d(self.depths_normalized, self.values, kind='cubic',
                            fill_value=fill_value,
                            bounds_error=(self.extrapolate == 'error'),
                            assume_sorted=True)

        elif self.method == 'spline':
            k = min(3, len(self.depths_normalized) - 1)
            s = self.smoothing if self.smoothing is not None else 0

            if self.extrapolate == 'zero':
                ext = 1
            elif self.extrapolate == 'constant':
                ext = 3
            elif self.extrapolate == 'error':
                ext = 2
            else:
                ext = 1

            return UnivariateSpline(self.depths_normalized, self.values, s=s, k=k, ext=ext)

        else:
            raise ValueError(f"Unknown interpolation method: {self.method}")

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self._interpolator(x)

    def evaluate_at_depths(self, depths_km: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        x_normalized = DepthCoordinateSystem.depth_to_normalized(depths_km)
        return self(x_normalized)

    def to_function(self, lebesgue_space):
        """
        Convert the interpolator into a Function object for the given
        Lebesgue space.

        Returns a Function that wraps the interpolator as an evaluate_callable.
        The sensitivity kernels are DATA, not basis expansions, so we don't
        project them onto the basis.

        The interpolator works in normalized coordinates [0,1], but the
        Lebesgue space may have a different domain (e.g., [0, 6371] km).
        We convert from the Lebesgue domain to normalized coordinates
        before evaluating the interpolator.
        """
        # Get the domain bounds from the Lebesgue space
        domain = lebesgue_space.function_domain
        a, b = domain.a, domain.b

        # Create a callable that converts from domain coordinates to normalized
        def kernel_callable(x):
            # Convert from [a, b] to [0, 1]
            x_normalized = (x - a) / (b - a)
            return self(x_normalized)

        # Return a Function object directly - no basis projection needed!
        return Function(lebesgue_space, evaluate_callable=kernel_callable)

    def integrate(self, x_min: float = 0.0, x_max: float = 1.0) -> float:
        from scipy.integrate import quad
        result, _ = quad(self, x_min, x_max)
        return result

    def get_peak_depth(self) -> Tuple[float, float]:
        x_sample = np.linspace(0, 1, 1000)
        y_sample = self(x_sample)
        abs_y = np.abs(y_sample)
        peak_idx = np.argmax(abs_y)
        peak_x = x_sample[peak_idx]
        peak_y = y_sample[peak_idx]
        peak_depth = DepthCoordinateSystem.normalized_to_depth(peak_x)
        return peak_depth, peak_y

    def plot(self, ax=None, n_points: int = 200, show_data: bool = True,
            use_depth: bool = True, **kwargs):
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        x_norm = np.linspace(0, 1, n_points)
        y_interp = self(x_norm)

        if use_depth:
            x_plot = DepthCoordinateSystem.normalized_to_depth(x_norm)
            x_data = self.depths_km
            xlabel = 'Depth [km]'
        else:
            x_plot = x_norm
            x_data = self.depths_normalized
            xlabel = 'Normalized Depth'

        label = kwargs.pop('label', f'{self.method} interpolation')
        ax.plot(x_plot, y_interp, label=label, **kwargs)

        if show_data:
            ax.scatter(x_data, self.values, color='red', s=20, zorder=5,
                       label='Data points', alpha=0.6)

        ax.set_xlabel(xlabel)
        ax.set_ylabel('Kernel Value')
        ax.grid(True, alpha=0.3)
        ax.legend()
        return ax

    def __repr__(self) -> str:
        return (
            f"KernelInterpolator(n_points={len(self.depths_km)}, method='{self.method}', "
            f"depth_range=[{self.depths_km.min():.1f}, {self.depths_km.max():.1f}] km)"
        )


def compare_interpolation_methods(depths_km: np.ndarray,
                                  values: np.ndarray,
                                  methods: Optional[list] = None,
                                  ax=None):
    import matplotlib.pyplot as plt
    if methods is None:
        methods = ['linear', 'cubic', 'spline']

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    interpolators = {}
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

    for method, color in zip(methods, colors):
        interp = KernelInterpolator(depths_km, values, method=method)
        interpolators[method] = interp
        x_norm = np.linspace(0, 1, 200)
        x_depth = DepthCoordinateSystem.normalized_to_depth(x_norm)
        y = interp(x_norm)
        ax.plot(x_depth, y, label=method, color=color, linewidth=2)

    ax.scatter(depths_km, values, color='black', s=50, zorder=10,
               label='Data', marker='o', edgecolors='white', linewidths=1.5)

    ax.set_xlabel('Depth [km]')
    ax.set_ylabel('Kernel Value')
    ax.set_title('Comparison of Interpolation Methods')
    ax.grid(True, alpha=0.3)
    ax.legend()

    return interpolators, ax
