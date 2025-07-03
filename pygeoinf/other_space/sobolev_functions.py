"""
Sobolev functions on interval domains.

This module provides function objects that live on IntervalDomain with
Sobolev regularity properties, designed to bridge the gap between
mathematical abstraction and computational representation.
"""

import numpy as np
from typing import Union, Callable, Optional
from .interval_domain import IntervalDomain
from .interval import Sobolev


class SobolevFunction:
    """
    A function in H^s([a,b]) with both mathematical and computational aspects.

    This class represents a function with Sobolev regularity that knows about
    the Sobolev space it belongs to. Functions can be defined via callable
    rules or basis representations.

    Note: Point evaluation is only well-defined for s > d/2 where s is the
    Sobolev order and d is the spatial dimension. For intervals (d=1),
    point evaluation requires s > 1/2.
    """

    def __init__(self,
                 space: Sobolev,
                 *,
                 coefficients: Optional[np.ndarray] = None,
                 evaluate_callable: Optional[Callable] = None,
                 name: Optional[str] = None):
        """
        Initialize a Sobolev function.

        Args:
            space: The Sobolev space this function belongs to
            coefficients: Optional finite-dimensional coefficient representation
            evaluate_callable: Optional callable defining the function rule
            name: Optional function name

        Note:
            Either coefficients or evaluate_callable must be provided.
        """
        self.space = space
        self.name = name
        # Function representation
        self.coefficients = coefficients.copy() if coefficients is not None else None
        self.evaluate_callable = evaluate_callable
        # Validate that we have a way to evaluate the function
        if self.coefficients is None and self.evaluate_callable is None:
            raise ValueError("Must provide either coefficients or evaluate_callable")

    @property
    def domain(self) -> IntervalDomain:
        """Get the IntervalDomain from the space."""
        return self.space.interval_domain

    @property
    def sobolev_order(self) -> float:
        """Get the Sobolev order from the space."""
        return self.space.order

    @property
    def boundary_conditions(self) -> Optional[dict]:
        """Get boundary conditions (if any)."""
        # The existing Sobolev class doesn't store boundary conditions explicitly
        return None

    def evaluate(self, x: Union[float, np.ndarray],
                 check_domain: bool = True) -> Union[float, np.ndarray]:
        """
        Point evaluation: f(x).

        Note: Point evaluation is only well-defined for Sobolev functions
        with s > d/2. For intervals (d=1), requires s > 1/2.

        Args:
            x: Point(s) at which to evaluate
            check_domain: Whether to check domain membership

        Returns:
            Function value(s)

        Raises:
            ValueError: If point evaluation is not well-defined for this Sobolev order
        """
        # Check mathematical validity of point evaluation
        if self.sobolev_order <= 0.5:  # d/2 = 1/2 for intervals
            raise ValueError(
                f"Point evaluation not well-defined for H^{self.sobolev_order} "
                f"on 1D domain. Requires s > 1/2."
            )

        # Check domain membership if requested
        if check_domain:
            x_array = np.asarray(x)
            if not np.all(self.domain.contains(x_array)):
                raise ValueError(f"Some points not in domain {self.domain}")

        # Evaluate using appropriate method
        if self.evaluate_callable is not None:
            return self.evaluate_callable(x)
        elif self.coefficients is not None:
            # Use the space's from_coefficient method to evaluate
            # Note: This assumes the space can evaluate at arbitrary points
            # For now, we'll use interpolation or direct evaluation if available
            return self._evaluate_from_coefficients(x)
        else:
            raise RuntimeError("No evaluation method available")

    def _evaluate_from_coefficients(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Helper method to evaluate from coefficients using the space's methods."""
        # Get function values on the space's grid
        try:
            # Get function values from coefficients
            function_values = self.space.from_coefficient(self.coefficients)

            # For proper evaluation, we need to know the grid points
            # This is a simplified implementation that assumes uniform grid
            # In practice, this would depend on the specific basis used

            # Create a uniform grid (should match the space's grid)
            x_grid = np.linspace(
                self.domain.a, self.domain.b, len(function_values)
            )

            # Interpolate to requested points
            x_array = np.asarray(x)
            is_scalar = x_array.ndim == 0
            if is_scalar:
                x_array = x_array.reshape(1)

            # Simple linear interpolation
            interpolated = np.interp(x_array, x_grid, function_values)

            return interpolated[0] if is_scalar else interpolated

        except Exception as e:
            raise NotImplementedError(
                "Point evaluation from coefficients not yet fully "
                f"implemented: {e}. Use evaluate_callable for now."
            )

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Allow f(x) syntax."""
        return self.evaluate(x)

    def integrate(self, weight: Optional[Callable] = None,
                 method: str = 'adaptive') -> float:
        """
        Integrate function over its domain: ∫[a,b] f(x) w(x) dx.

        Args:
            weight: Optional weight function w(x)
            method: Integration method

        Returns:
            Integral value
        """
        if weight is None:
            # Direct integration based on representation
            if self.evaluate_callable is not None:
                return self.domain.integrate(self.evaluate_callable, method=method)
            elif self.coefficients is not None:
                # For basis representations, might have analytical formulas
                integrand = lambda x: self.evaluate(x, check_domain=False)
                return self.domain.integrate(integrand, method=method)
        else:
            def integrand(x):
                return self.evaluate(x, check_domain=False) * weight(x)
            return self.domain.integrate(integrand, method=method)

    # Inner product and norm are now only in the Sobolev space class

    def restrict_to(self, subdomain: IntervalDomain) -> 'SobolevFunction':
        """
        Restriction to subdomain.

        Args:
            subdomain: Target subdomain

        Returns:
            Restricted function
        """
        # This is a simplified implementation
        # In practice, you'd need to handle the basis transformation properly
        # Create a new Sobolev space for the subdomain (simplified)
        new_space = Sobolev.create_standard_sobolev(
            order=self.sobolev_order,
            scale=0.1,  # Default scale - should be parameterized
            dim=self.space.dim,
            interval=(subdomain.a, subdomain.b)
        )

        return SobolevFunction(
            new_space,
            coefficients=self.coefficients.copy() if self.coefficients is not None else None,
            evaluate_callable=self.evaluate_callable
        )

    def extend_to(self, larger_domain: IntervalDomain,
                 method: str = 'zero') -> 'SobolevFunction':
        """
        Extension to larger domain.

        Args:
            larger_domain: Target larger domain
            method: Extension method ('zero', 'constant', 'reflection')

        Returns:
            Extended function
        """
        if not (larger_domain.a <= self.domain.a and
                self.domain.b <= larger_domain.b):
            raise ValueError("Target domain must contain current domain")

        # Simplified implementation - proper extension depends on basis
        # Create a new Sobolev space for the larger domain
        new_space = Sobolev.create_standard_sobolev(
            order=self.sobolev_order,
            scale=0.1,  # Default scale - should be parameterized
            dim=self.space.dim,
            interval=(larger_domain.a, larger_domain.b)
        )

        return SobolevFunction(
            new_space,
            coefficients=self.coefficients.copy() if self.coefficients is not None else None,
            evaluate_callable=self.evaluate_callable
        )

    def weak_derivative(self, order: int = 1) -> 'SobolevFunction':
        """
        Compute weak derivative of given order.

        Args:
            order: Derivative order

        Returns:
            Derivative as Sobolev function
        """
        if order > self.sobolev_order:
            raise ValueError(
                f"Cannot take derivative of order {order} "
                f"for H^{self.sobolev_order} function"
            )

        # This is simplified - would need proper implementation based on basis
        if self.coefficients is not None:
            # For now, just return a copy with reduced order
            return SobolevFunction(
                self.space,
                coefficients=self.coefficients.copy()
            )
        else:
            raise NotImplementedError(
                "Weak derivative for callable functions not implemented"
            )

    def plot(self, n_points: int = 1000, **kwargs):
        """
        Plot the function.

        Args:
            n_points: Number of plot points
            **kwargs: Additional plotting arguments
        """
        import matplotlib.pyplot as plt

        x = self.domain.uniform_mesh(n_points)
        y = self.evaluate(x)

        plt.plot(x, y, label=self.name or "Sobolev function", **kwargs)
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title(f'Function on {self.domain}')
        if self.name:
            plt.legend()
        plt.grid(True, alpha=0.3)

    def __add__(self, other):
        """Addition of Sobolev functions."""
        if isinstance(other, SobolevFunction):
            if other.space != self.space:
                raise ValueError("Functions must be in the same Sobolev space")

            if (self.coefficients is not None and
                    other.coefficients is not None):
                new_coeffs = self.coefficients + other.coefficients
                return SobolevFunction(
                    self.space, coefficients=new_coeffs
                )
            else:
                # For callable functions, create a new callable
                def new_callable(x):
                    return self.evaluate(x) + other.evaluate(x)
                return SobolevFunction(
                    self.space, evaluate_callable=new_callable
                )
        else:
            # Adding a constant
            if self.coefficients is not None:
                new_coeffs = self.coefficients.copy()
                new_coeffs[0] += other  # Assume first coefficient is constant
                return SobolevFunction(
                    self.space, coefficients=new_coeffs
                )
            else:
                def new_callable(x):
                    return self.evaluate(x) + other
                return SobolevFunction(
                    self.space, evaluate_callable=new_callable
                )

    def __mul__(self, other):
        """Multiplication of Sobolev functions or by scalars."""
        if isinstance(other, (int, float)):
            # Scalar multiplication
            if self.coefficients is not None:
                return SobolevFunction(
                    self.space,
                    coefficients=other * self.coefficients
                )
            else:
                def new_callable(x):
                    return other * self.evaluate(x)
                return SobolevFunction(
                    self.space,
                    evaluate_callable=new_callable
                )
        elif isinstance(other, SobolevFunction):
            # Function multiplication - complex, depends on basis
            raise NotImplementedError(
                "Function multiplication not yet implemented"
            )
        else:
            raise TypeError(
                f"Cannot multiply SobolevFunction with {type(other)}"
            )

    def __rmul__(self, other):
        """Right multiplication (for scalar * function)."""
        return self.__mul__(other)

    def __repr__(self) -> str:
        return (f"SobolevFunction(domain={self.domain}, "
                f"order={self.sobolev_order}, name={self.name})")


def create_sobolev_function(space: Sobolev,
                            *,
                            coefficients: Optional[np.ndarray] = None,
                            evaluate_callable: Optional[Callable] = None,
                            **kwargs) -> SobolevFunction:
    """
    Factory function to create Sobolev functions in an existing Sobolev space.

    Args:
        space: Existing Sobolev space from interval.py
        coefficients: Optional coefficient representation
        evaluate_callable: Optional callable defining function rule
        **kwargs: Additional arguments

    Returns:
        SobolevFunction instance

    Examples:
        # Create Sobolev space first
        space = Sobolev.create_standard_sobolev(
            order=1.5, scale=0.1, dim=50, interval=(0, np.pi)
        )

        # Using a callable (like SOLA_DLI approach)
        f = create_sobolev_function(
            space,
            evaluate_callable=lambda x: x**2 * np.sin(x)
        )

        # Using basis coefficients
        coeffs = np.random.randn(space.dim) * np.exp(
            -np.arange(space.dim) * 0.1
        )
        f = create_sobolev_function(
            space,
            coefficients=coeffs
        )
    """
    return SobolevFunction(
        space,
        coefficients=coefficients,
        evaluate_callable=evaluate_callable,
        **kwargs
    )
