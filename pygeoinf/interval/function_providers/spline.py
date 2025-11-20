"""Spline function provider."""

import numpy as np
from typing import Dict, Any
from .base import IndexedFunctionProvider, ParametricFunctionProvider


class SplineFunctionProvider(IndexedFunctionProvider,
                             ParametricFunctionProvider):
    """Provider for spline functions."""

    def __init__(self, space):
        """
        Initialize spline provider.

        Args:
            space: Lebesgue instance (contains domain information)
        """
        super().__init__(space)

    def get_function_by_index(self, index: int, degree: int = 3,
                              n_knots: int = 10, **kwargs) -> 'Function':
        """Get B-spline basis function by index."""
        from ..functions import Function
        from scipy.interpolate import BSpline

        a, b = self.domain.a, self.domain.b

        # Create knot vector
        internal_knots = np.linspace(a, b, n_knots + 2)[1:-1]
        knots = np.concatenate([
            [a] * (degree + 1),
            internal_knots,
            [b] * (degree + 1)
        ])

        # Create coefficient vector (1 at index, 0 elsewhere)
        n_coeffs = len(knots) - degree - 1
        coeffs = np.zeros(n_coeffs)
        coeffs[index % n_coeffs] = 1.0

        spline = BSpline(knots, coeffs, degree)

        def spline_func(x):
            return spline(x)

        return Function(
            self.space,
            evaluate_callable=spline_func,
            name=f'spline_{index}'
        )

    def get_function_by_parameters(self, parameters: Dict[str, Any],
                                   **kwargs) -> 'Function':
        """Get spline with specific parameters."""
        from ..functions import Function
        from scipy.interpolate import BSpline

        a, b = self.domain.a, self.domain.b
        degree = parameters.get('degree', 3)
        knots = parameters.get('knots', np.linspace(a, b, 10))
        coeffs = parameters.get('coeffs', np.ones(len(knots) - degree - 1))

        spline = BSpline(knots, coeffs, degree)

        def spline_func(x):
            return spline(x)

        return Function(
            self.space,
            evaluate_callable=spline_func,
            name=f'spline_deg{degree}'
        )

    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default spline parameters."""
        a, b = self.domain.a, self.domain.b
        return {
            'degree': 3,
            'knots': np.linspace(a, b, 10),
            'coeffs': np.ones(6)  # Compatible with degree 3 and 10 knots
        }
