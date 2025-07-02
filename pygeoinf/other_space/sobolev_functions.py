"""
Sobolev functions on interval domains.

This module provides function objects that live on IntervalDomain with
Sobolev regularity properties, designed to bridge the gap between
mathematical abstraction and computational representation.
"""

import numpy as np
from typing import Union, Callable, Optional, Any
from .interval_domain import IntervalDomain, BoundaryConditions


class SobolevFunction:
    """
    A function in H^s([a,b]) with both mathematical and computational aspects.

    This class represents a function with Sobolev regularity that knows about
    its domain structure while maintaining computational efficiency through
    basis representations.
    """

    def __init__(self,
                 domain: IntervalDomain,
                 coefficients: np.ndarray,
                 basis_transform: 'BasisTransform',
                 sobolev_order: float,
                 *,
                 boundary_conditions: Optional[dict] = None,
                 name: Optional[str] = None):
        """
        Initialize a Sobolev function.

        Args:
            domain: The interval domain
            coefficients: Finite-dimensional coefficient representation
            basis_transform: Object handling basis transformations
            sobolev_order: Sobolev regularity order s in H^s
            boundary_conditions: Boundary condition specification
            name: Optional function name
        """
        self.domain = domain
        self.coefficients = coefficients.copy()
        self.basis_transform = basis_transform
        self.sobolev_order = sobolev_order
        self.boundary_conditions = boundary_conditions
        self.name = name

    def evaluate(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Point evaluation: f(x).

        Args:
            x: Point(s) at which to evaluate

        Returns:
            Function value(s)
        """
        # Check domain membership
        x_array = np.asarray(x)
        if not np.all(self.domain.contains(x_array)):
            raise ValueError(f"Some points not in domain {self.domain}")

        return self.basis_transform.from_coefficients(self.coefficients, x)

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Allow f(x) syntax."""
        return self.evaluate(x)

    def integrate(self, weight: Optional[Callable] = None, method: str = 'adaptive') -> float:
        """
        Integrate function over its domain: ∫[a,b] f(x) w(x) dx.

        Args:
            weight: Optional weight function w(x)
            method: Integration method

        Returns:
            Integral value
        """
        if weight is None:
            integrand = self.evaluate
        else:
            def integrand(x):
                return self.evaluate(x) * weight(x)

        return self.domain.integrate(integrand, method=method)

    def inner_product(self, other: 'SobolevFunction', order: Optional[float] = None) -> float:
        """
        Sobolev inner product with another function.

        Args:
            other: Another Sobolev function
            order: Sobolev order (defaults to self.sobolev_order)

        Returns:
            Inner product value
        """
        if other.domain != self.domain:
            raise ValueError("Functions must be on the same domain")

        order = order or self.sobolev_order

        # This would need to be implemented based on the specific basis
        # For now, use the coefficient-based inner product
        if hasattr(self.basis_transform, 'sobolev_inner_product'):
            return self.basis_transform.sobolev_inner_product(
                self.coefficients, other.coefficients, order
            )
        else:
            # Fallback to L2 inner product
            def integrand(x):
                return self.evaluate(x) * other.evaluate(x)
            return self.domain.integrate(integrand)

    def norm(self, order: Optional[float] = None) -> float:
        """
        Sobolev norm of the function.

        Args:
            order: Sobolev order (defaults to self.sobolev_order)

        Returns:
            Norm value
        """
        return np.sqrt(self.inner_product(self, order))

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
        return SobolevFunction(
            subdomain,
            self.coefficients.copy(),  # Simplified - may need projection
            self.basis_transform,
            self.sobolev_order,
            boundary_conditions=self.boundary_conditions
        )

    def extend_to(self, larger_domain: IntervalDomain, method: str = 'zero') -> 'SobolevFunction':
        """
        Extension to larger domain.

        Args:
            larger_domain: Target larger domain
            method: Extension method ('zero', 'constant', 'reflection')

        Returns:
            Extended function
        """
        if not (larger_domain.a <= self.domain.a and self.domain.b <= larger_domain.b):
            raise ValueError("Target domain must contain current domain")

        # Simplified implementation - proper extension depends on basis and method
        # This would need more sophisticated implementation
        return SobolevFunction(
            larger_domain,
            self.coefficients.copy(),
            self.basis_transform,
            self.sobolev_order
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
            raise ValueError(f"Cannot take derivative of order {order} for H^{self.sobolev_order} function")

        # This depends on the specific basis
        derivative_coeffs = self.basis_transform.differentiate_coefficients(
            self.coefficients, order
        )

        return SobolevFunction(
            self.domain,
            derivative_coeffs,
            self.basis_transform,
            self.sobolev_order - order
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
            if other.domain != self.domain:
                raise ValueError("Functions must be on the same domain")
            new_coeffs = self.coefficients + other.coefficients
            min_order = min(self.sobolev_order, other.sobolev_order)
        else:
            # Adding a constant
            new_coeffs = self.coefficients.copy()
            new_coeffs[0] += other  # Assume first coefficient is constant term
            min_order = self.sobolev_order

        return SobolevFunction(
            self.domain, new_coeffs, self.basis_transform, min_order
        )

    def __mul__(self, other):
        """Multiplication of Sobolev functions or by scalars."""
        if isinstance(other, (int, float)):
            # Scalar multiplication
            return SobolevFunction(
                self.domain,
                other * self.coefficients,
                self.basis_transform,
                self.sobolev_order
            )
        elif isinstance(other, SobolevFunction):
            # Function multiplication - this is more complex and depends on basis
            raise NotImplementedError("Function multiplication not yet implemented")
        else:
            raise TypeError(f"Cannot multiply SobolevFunction with {type(other)}")

    def __rmul__(self, other):
        """Right multiplication (for scalar * function)."""
        return self.__mul__(other)

    def __repr__(self) -> str:
        return f"SobolevFunction(domain={self.domain}, order={self.sobolev_order}, name={self.name})"


class BasisTransform:
    """
    Abstract base class for basis transformations.

    This handles the conversion between function values and coefficient
    representations in various bases (Fourier, polynomial, etc.).
    """

    def __init__(self, domain: IntervalDomain):
        self.domain = domain

    def to_coefficients(self, function_values: np.ndarray) -> np.ndarray:
        """Transform function values to coefficients."""
        raise NotImplementedError

    def from_coefficients(self, coefficients: np.ndarray,
                         x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate function from coefficients at points x."""
        raise NotImplementedError

    def differentiate_coefficients(self, coefficients: np.ndarray, order: int) -> np.ndarray:
        """Compute coefficients of derivative."""
        raise NotImplementedError

    def sobolev_inner_product(self, coeffs1: np.ndarray, coeffs2: np.ndarray,
                            order: float) -> float:
        """Compute Sobolev inner product using coefficients."""
        raise NotImplementedError


class FourierBasisTransform(BasisTransform):
    """
    Fourier (DCT) basis transformation for Neumann boundary conditions.
    """

    def to_coefficients(self, function_values: np.ndarray) -> np.ndarray:
        """DCT transformation."""
        from scipy.fft import dct
        return dct(function_values, type=2, norm='ortho')

    def from_coefficients(self, coefficients: np.ndarray,
                         x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate Fourier series at points x."""
        # This is a simplified implementation
        # Proper implementation would evaluate the cosine series directly
        from scipy.fft import idct

        # For evaluation at arbitrary points, we'd need to implement
        # the actual cosine series evaluation
        # This is a placeholder that assumes uniform grid evaluation
        n = len(coefficients)
        x_grid = self.domain.uniform_mesh(n)
        f_grid = idct(coefficients, type=2, norm='ortho')

        # Interpolate to get values at x
        return np.interp(x, x_grid, f_grid)

    def differentiate_coefficients(self, coefficients: np.ndarray, order: int) -> np.ndarray:
        """Differentiate in Fourier space."""
        n = len(coefficients)
        length = self.domain.length

        # Frequency multipliers for derivatives
        freqs = np.arange(n) * np.pi / length
        multiplier = (1j * freqs) ** order

        # For real DCT, derivatives are more complex
        # This is a simplified implementation
        derivative_coeffs = coefficients.copy()
        for k in range(1, n):
            derivative_coeffs[k] *= (k * np.pi / length) ** order
            if order % 2 == 1:
                derivative_coeffs[k] *= -1 if k % 2 == 1 else 1

        return derivative_coeffs


def create_sobolev_function(domain: IntervalDomain,
                          coefficients: np.ndarray,
                          basis_type: str = 'fourier',
                          sobolev_order: float = 1.0,
                          **kwargs) -> SobolevFunction:
    """
    Factory function to create Sobolev functions with different basis types.

    Args:
        domain: Interval domain
        coefficients: Coefficient representation
        basis_type: Type of basis ('fourier', 'polynomial', etc.)
        sobolev_order: Sobolev regularity order
        **kwargs: Additional arguments

    Returns:
        SobolevFunction instance
    """
    if basis_type == 'fourier':
        basis_transform = FourierBasisTransform(domain)
    else:
        raise ValueError(f"Unknown basis type: {basis_type}")

    return SobolevFunction(
        domain, coefficients, basis_transform, sobolev_order, **kwargs
    )
