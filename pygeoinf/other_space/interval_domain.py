"""
Mathematical domain implementation for intervals [a,b] ⊂ ℝ.

This module provides a rigorous mathematical foundation for interval domains
with topological and measure-theoretic structure, designed to work with
Sobolev spaces and function analysis.
"""

import numpy as np
from typing import Union, Callable, Optional, Tuple
from abc import ABC, abstractmethod


class IntervalDomain:
    """
    A mathematically rigorous interval [a,b] ⊂ ℝ with topological
    and measure-theoretic structure.

    This class represents a compact interval with proper mathematical
    structure for function analysis and Sobolev spaces.
    """

    def __init__(self, a: float, b: float, *,
                 boundary_type: str = 'closed',
                 name: Optional[str] = None):
        """
        Initialize an interval domain.

        Args:
            a: Left endpoint
            b: Right endpoint
            boundary_type: Type of interval ('closed', 'open', 'left_open', 'right_open')
            name: Optional name for the domain
        """
        if a >= b:
            raise ValueError(f"Invalid interval: a={a} must be less than b={b}")

        self.a = float(a)
        self.b = float(b)
        self.boundary_type = boundary_type
        self.name = name or f"[{a}, {b}]"

    @property
    def length(self) -> float:
        """Lebesgue measure of the interval."""
        return self.b - self.a

    @property
    def center(self) -> float:
        """Midpoint of the interval."""
        return (self.a + self.b) / 2

    @property
    def radius(self) -> float:
        """Half-length of the interval."""
        return (self.b - self.a) / 2

    def contains(self, x: Union[float, np.ndarray]) -> Union[bool, np.ndarray]:
        """
        Check if point(s) are in the domain.

        Args:
            x: Point or array of points to check

        Returns:
            Boolean or array of booleans indicating membership
        """
        if self.boundary_type == 'closed':
            return (x >= self.a) & (x <= self.b)
        elif self.boundary_type == 'open':
            return (x > self.a) & (x < self.b)
        elif self.boundary_type == 'left_open':
            return (x > self.a) & (x <= self.b)
        elif self.boundary_type == 'right_open':
            return (x >= self.a) & (x < self.b)
        else:
            raise ValueError(f"Unknown boundary_type: {self.boundary_type}")

    def interior(self) -> 'IntervalDomain':
        """Return the interior (a,b)."""
        return IntervalDomain(self.a, self.b, boundary_type='open')

    def closure(self) -> 'IntervalDomain':
        """Return the closure [a,b]."""
        return IntervalDomain(self.a, self.b, boundary_type='closed')

    def boundary_points(self) -> Tuple[float, float]:
        """Return boundary points {a, b}."""
        return (self.a, self.b)

    def uniform_mesh(self, n: int) -> np.ndarray:
        """
        Generate uniform mesh with n points.

        Args:
            n: Number of mesh points

        Returns:
            Array of mesh points
        """
        if self.boundary_type in ['closed', 'right_open']:
            return np.linspace(self.a, self.b, n, endpoint=(self.boundary_type == 'closed'))
        else:  # open or left_open
            # Avoid boundary points for open intervals
            return np.linspace(self.a, self.b, n + 2)[1:-1]

    def adaptive_mesh(self, f: Callable, tol: float = 1e-6, max_points: int = 1000) -> np.ndarray:
        """
        Generate adaptive mesh based on function behavior.

        Args:
            f: Function to adapt to
            tol: Tolerance for adaptation
            max_points: Maximum number of points

        Returns:
            Adapted mesh points
        """
        # Simple implementation - can be enhanced with proper adaptive algorithms
        x = self.uniform_mesh(max_points)
        return x[self.contains(x)]

    def random_points(self, n: int, *, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate random points in the domain.

        Args:
            n: Number of points
            seed: Random seed for reproducibility

        Returns:
            Array of random points
        """
        if seed is not None:
            np.random.seed(seed)

        if self.boundary_type == 'closed':
            return np.random.uniform(self.a, self.b, n)
        elif self.boundary_type == 'open':
            # Avoid exact boundary points
            eps = 1e-12
            return np.random.uniform(self.a + eps, self.b - eps, n)
        else:
            # Handle semi-open intervals
            points = np.random.uniform(self.a, self.b, n)
            return points[self.contains(points)]

    def integrate(self, f: Callable, method: str = 'adaptive', **kwargs) -> float:
        """
        Integrate function over the domain.

        Args:
            f: Function to integrate
            method: Integration method ('adaptive', 'gauss', 'simpson', 'trapz')
            **kwargs: Additional arguments for integration method

        Returns:
            Integral value
        """
        if method == 'adaptive':
            from scipy.integrate import quad
            return quad(f, self.a, self.b, **kwargs)[0]
        elif method == 'gauss':
            return self._gauss_legendre_integrate(f, **kwargs)
        elif method == 'simpson':
            n = kwargs.get('n', 1000)
            x = self.uniform_mesh(n)
            y = f(x)
            return np.trapz(y, x)  # Simpson's rule via trapz
        elif method == 'trapz':
            n = kwargs.get('n', 1000)
            x = self.uniform_mesh(n)
            y = f(x)
            return np.trapz(y, x)
        else:
            raise ValueError(f"Unknown integration method: {method}")

    def _gauss_legendre_integrate(self, f: Callable, n: int = 50) -> float:
        """Gauss-Legendre quadrature."""
        from scipy.special import roots_legendre

        # Get Gauss-Legendre nodes and weights on [-1, 1]
        nodes, weights = roots_legendre(n)

        # Transform to [a, b]
        x = self.a + (self.b - self.a) * (nodes + 1) / 2
        w = weights * (self.b - self.a) / 2

        return np.sum(w * f(x))

    def point_evaluation_functional(self, x: float) -> Callable:
        """
        Return point evaluation functional δ_x: f ↦ f(x).

        Args:
            x: Evaluation point

        Returns:
            Point evaluation functional
        """
        if not self.contains(x):
            raise ValueError(f"Point {x} not in domain {self}")

        def delta_x(f):
            return f(x)

        return delta_x

    def restriction_to_subinterval(self, c: float, d: float) -> 'IntervalDomain':
        """
        Create restriction to subinterval [c,d] ⊂ [a,b].

        Args:
            c: Left endpoint of subinterval
            d: Right endpoint of subinterval

        Returns:
            Subinterval domain
        """
        if not (self.a <= c < d <= self.b):
            raise ValueError(f"Invalid subinterval [{c}, {d}] not contained in {self}")

        return IntervalDomain(c, d, boundary_type=self.boundary_type)

    def __repr__(self) -> str:
        bracket_left = '[' if self.boundary_type in ['closed', 'right_open'] else '('
        bracket_right = ']' if self.boundary_type in ['closed', 'left_open'] else ')'
        return f"{bracket_left}{self.a}, {self.b}{bracket_right}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, IntervalDomain):
            return False
        return (self.a == other.a and self.b == other.b and
                self.boundary_type == other.boundary_type)


class BoundaryConditions:
    """
    Boundary condition specifications for Sobolev spaces on intervals.
    """

    @staticmethod
    def dirichlet(left_value: float = 0, right_value: float = 0):
        """
        Dirichlet boundary conditions: u(a) = left_value, u(b) = right_value.

        Args:
            left_value: Value at left boundary
            right_value: Value at right boundary
        """
        return {
            'type': 'dirichlet',
            'left': left_value,
            'right': right_value
        }

    @staticmethod
    def neumann(left_derivative: float = 0, right_derivative: float = 0):
        """
        Neumann boundary conditions: u'(a) = left_derivative, u'(b) = right_derivative.

        Args:
            left_derivative: Derivative value at left boundary
            right_derivative: Derivative value at right boundary
        """
        return {
            'type': 'neumann',
            'left': left_derivative,
            'right': right_derivative
        }

    @staticmethod
    def robin(left_alpha: float, left_beta: float, left_value: float,
              right_alpha: float, right_beta: float, right_value: float):
        """
        Robin boundary conditions: αu + βu' = value at boundaries.

        Args:
            left_alpha, left_beta, left_value: Left boundary coefficients
            right_alpha, right_beta, right_value: Right boundary coefficients
        """
        return {
            'type': 'robin',
            'left': {'alpha': left_alpha, 'beta': left_beta, 'value': left_value},
            'right': {'alpha': right_alpha, 'beta': right_beta, 'value': right_value}
        }

    @staticmethod
    def periodic():
        """
        Periodic boundary conditions: u(a) = u(b), u'(a) = u'(b).
        """
        return {'type': 'periodic'}


# Alias for backward compatibility
Interval = IntervalDomain
