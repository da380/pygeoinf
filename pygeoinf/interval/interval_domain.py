"""
Mathematical domain implementation for intervals [a,b] ⊂ ℝ.

This module provides a rigorous mathematical foundation for interval domains
with topological and measure-theoretic structure, designed to work with
Sobolev spaces and function analysis.
"""

import numpy as np
from typing import Union, Callable, Optional, Tuple


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
            boundary_type: Type of interval
                ('closed', 'open', 'left_open', 'right_open')
            name: Optional name for the domain
        """
        if a >= b:
            raise ValueError(
                f"Invalid interval: a={a} must be less than b={b}"
            )

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
        if self.boundary_type == 'closed':
            return np.linspace(self.a, self.b, n, endpoint=True)
        elif self.boundary_type == 'open':
            # Exclude both endpoints
            return np.linspace(self.a, self.b, n + 2)[1:-1]
        elif self.boundary_type == 'left_open':
            # Exclude left endpoint, include right endpoint
            return np.linspace(self.a, self.b, n + 1)[1:]
        elif self.boundary_type == 'right_open':
            # Include left endpoint, exclude right endpoint
            return np.linspace(self.a, self.b, n, endpoint=False)
        else:
            raise ValueError(f"Unknown boundary_type: {self.boundary_type}")

    def adaptive_mesh(
        self,
        f: Callable,
        tol: float = 1e-6,
        max_points: int = 1000
    ) -> np.ndarray:
        """
        Generate adaptive mesh based on function behavior.

        Args:
            f: Function to adapt to
            tol: Tolerance for adaptation
            max_points: Maximum number of points

        Returns:
            Adapted mesh points
        """
        # Simple implementation - can be enhanced with proper adaptive
        # algorithms
        x = self.uniform_mesh(max_points)
        return x[self.contains(x)]

    def random_points(
        self, n: int, *, seed: Optional[int] = None
    ) -> np.ndarray:
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

    def integrate(
        self,
        f: Callable,
        method: str = 'adaptive',
        support: Optional[Union[Tuple[float, float],
                                "list[Tuple[float, float]]"]] = None,
        n_points: int = 100,
        **kwargs
    ) -> float:
        """
        Integrate function over the domain or subinterval(s).

        This method supports efficient integration over multiple disjoint
        subintervals, making it ideal for functions with compact support.

        Args:
            f: Function to integrate
            method: Integration method
                ('adaptive', 'simpson', 'trapz')
            subinterval: Optional integration bounds. Can be:
                - None: integrate over entire domain [self.a, self.b]
                - (a, b): integrate over single subinterval [a, b]
                - [(a1, b1), (a2, b2), ...]: integrate over multiple
                  disjoint subintervals (for compact support functions)
            n_points: Number of points for 'simpson' and 'trapz' methods
            **kwargs: Additional arguments for integration method
                - n: number of points for 'simpson' and 'trapz' methods
                - other scipy.integrate parameters for 'adaptive' method

        Returns:
            Integral value

        Examples:
            >>> domain = IntervalDomain(0.0, 2.0)
            >>> f = lambda x: x**2
            >>> # Full domain integration
            >>> domain.integrate(f)
            >>> # Single subinterval
            >>> domain.integrate(f, subinterval=(0.5, 1.5))
            >>> # Multiple subintervals (compact support)
            >>> domain.integrate(f, subinterval=[(0.2, 0.8), (1.2, 1.8)])
        """
        # Handle multiple subintervals (compact support)
        if support is not None and isinstance(support, list):
            total_integral = 0.0
            for a, b in support:
                if not (self.a <= a < b <= self.b):
                    raise ValueError(
                        f"support [{a}, {b}] not contained in "
                        f"domain [{self.a}, {self.b}]"
                    )
                # Recursively integrate over each subinterval
                integral_part = self.integrate(f, method=method,
                                               support=(a, b), **kwargs)
                total_integral += integral_part
            return total_integral

        # Handle single subinterval or full domain
        if support is not None:
            a, b = support
            if not (self.a <= a < b <= self.b):
                raise ValueError(
                    f"Support [{a}, {b}] not contained in "
                    f"domain [{self.a}, {self.b}]"
                )
        else:
            a, b = self.a, self.b
        # Update number of n_points based on length of the interval
        n_points_interval = max(3, int(n_points * (b - a) / (self.b - self.a)))

        if method == 'adaptive':
            try:
                from scipy.integrate import quad
                return quad(f, a, b, **kwargs)[0]
            except ImportError:
                raise ImportError(
                    "scipy is required for adaptive integration. "
                    "Install with: pip install scipy"
                )
        elif method == 'simpson':
            try:
                from scipy.integrate import simpson
                x = np.linspace(a, b, n_points_interval)
                y = self._evaluate_function_vectorized(f, x)
                return float(simpson(y, x=x))
            except ImportError:
                # Fallback to numpy trapz
                x = np.linspace(a, b, n_points_interval)
                y = self._evaluate_function_vectorized(f, x)
                return float(np.trapz(y, x=x))
        elif method == 'trapz':
            try:
                from scipy.integrate import trapezoid
                x = np.linspace(a, b, n_points_interval)
                y = self._evaluate_function_vectorized(f, x)
                return float(trapezoid(y, x=x))
            except ImportError:
                # Fallback to numpy trapz
                x = np.linspace(a, b, n_points_interval)
                y = self._evaluate_function_vectorized(f, x)
                return float(np.trapz(y, x=x))
        else:
            raise ValueError(f"Unknown integration method: {method}")

    def _evaluate_function_vectorized(
        self, f: Callable, x: np.ndarray
    ) -> np.ndarray:
        """
        Evaluate function f at array of points x, with vectorization
        optimization.

        Args:
            f: Function to evaluate
            x: Array of evaluation points

        Returns:
            Array of function values
        """
        try:
            # Try vectorized evaluation first
            result = f(x)
            # Check if result is scalar (function isn't vectorized) or array
            if np.isscalar(result):
                # Function isn't vectorized, fall back to loop
                return np.array([f(xi) for xi in x])
            else:
                # Function is vectorized
                return np.asarray(result)
        except (TypeError, ValueError):
            # Function doesn't support vectorized evaluation, use loop
            return np.array([f(xi) for xi in x])

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

    def restriction_to_subinterval(
        self, c: float, d: float
    ) -> 'IntervalDomain':
        """
        Create restriction to subinterval [c,d] ⊂ [a,b].

        Args:
            c: Left endpoint of subinterval
            d: Right endpoint of subinterval

        Returns:
            Subinterval domain
        """
        if not (self.a <= c < d <= self.b):
            raise ValueError(
                f"Invalid subinterval [{c}, {d}] not contained in {self}"
            )

        return IntervalDomain(c, d, boundary_type=self.boundary_type)

    def __repr__(self) -> str:
        if self.boundary_type in ['closed', 'right_open']:
            bracket_left = '['
        else:
            bracket_left = '('

        if self.boundary_type in ['closed', 'left_open']:
            bracket_right = ']'
        else:
            bracket_right = ')'

        return f"{bracket_left}{self.a}, {self.b}{bracket_right}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, IntervalDomain):
            return False
        return (self.a == other.a and self.b == other.b and
                self.boundary_type == other.boundary_type)
