"""
L² functions on interval domains.

This module provides function objects that live on IntervalDomain in L² spaces,
serving as the base class for more specialized function spaces
like Sobolev spaces.
"""

import numpy as np
from typing import Union, Callable, Optional
import numbers


class L2Function:
    """
    A function in L²([a,b]) with both mathematical and computational aspects.

    This class represents a function in L² space that knows about the L² space
    it belongs to. Functions can be defined via callable rules or basis
    representations.

    This serves as the base class for more specialized function spaces.
    """

    def __init__(self,
                 space,
                 *,
                 coefficients: Optional[np.ndarray] = None,
                 evaluate_callable: Optional[Callable] = None,
                 name: Optional[str] = None):
        """
        Initialize an L² function.

        Args:
            space: The L²Space this function belongs to
            coefficients: Optional finite-dimensional coefficient
                representation
            evaluate_callable: Optional callable defining the function rule
            name: Optional function name

        Note:
            Either coefficients or evaluate_callable must be provided.
        """
        self.space = space
        self.name = name
        # Function representation
        self.coefficients = (coefficients.copy()
                             if coefficients is not None else None)
        self.evaluate_callable = evaluate_callable

        # Validate that we have a way to evaluate the function
        if self.coefficients is None and self.evaluate_callable is None:
            raise ValueError("Must provide either coefficients or "
                             "evaluate_callable")

    @property
    def domain(self):
        """Get the IntervalDomain from the space."""
        return self.space.interval_domain

    @property
    def space_type(self):
        """Get the type of space this function belongs to."""
        return "L2"

    def evaluate(self, x: Union[float, np.ndarray],
                 check_domain: bool = True) -> Union[float, np.ndarray]:
        """
        Point evaluation of representative: f(x).

        WARNING: Point evaluation is not mathematically well-defined for
        general L² functions, which are only defined almost everywhere.
        Elements of L² spaces are equivalence classes of functions
        that are equal almost everywhere, so pointwise evaluation may not
        yield meaningful results. However, some equivalence classes have
        a useful representative, and this method can be used to
        evaluate such representatives.

        Args:
            x: Point(s) at which to evaluate
            check_domain: Whether to check domain membership

        Returns:
            Function value(s)

        Note:
            This should be overridden in subclasses where point evaluation
            is mathematically justified (e.g., Sobolev functions with s > 1/2).
        """
        import warnings
        warnings.warn(
            "Point evaluation is not well-defined for general L² functions. "
            "Consider using a Sobolev space with s > 1/2 for point "
            "evaluation.",
            UserWarning
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
            return self._evaluate_from_coefficients(x)
        else:
            raise RuntimeError("No evaluation method available")

    def _evaluate_from_coefficients(
        self, x: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Evaluate the function at x using the basis and coefficients."""
        # Get the basis functions from the parent space
        basis_functions = self.space.basis_functions
        coeffs = self.coefficients
        if coeffs is None or basis_functions is None:
            raise RuntimeError("Coefficients or basis functions not available "
                               "for evaluation.")
        if len(coeffs) != len(basis_functions):
            raise ValueError(
                f"Coefficient length {len(coeffs)} does not match "
                f"number of basis functions {len(basis_functions)}."
            )

        x_array = np.asarray(x)
        is_scalar = x_array.ndim == 0
        if is_scalar:
            x_array = x_array.reshape(1)

        # Evaluate each basis function at x_array
        basis_evals = np.array([bf.evaluate(x_array, check_domain=False)
                                for bf in basis_functions])
        # basis_evals shape: (n_basis, n_points)
        # Linear combination: sum_k c_k * phi_k(x)
        result = np.tensordot(coeffs, basis_evals, axes=([0], [0]))
        return result[0] if is_scalar else result

    def __call__(
        self, x: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
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
                return self.domain.integrate(self.evaluate_callable,
                                             method=method)
            elif self.coefficients is not None:
                # For basis representations, might have analytical formulas
                def integrand(x):
                    return self.evaluate(x, check_domain=False)
                return self.domain.integrate(integrand, method=method)
        else:
            def integrand(x):
                return self.evaluate(x, check_domain=False) * weight(x)
            return self.domain.integrate(integrand, method=method)

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

        plt.plot(x, y, label=self.name or "L² function", **kwargs)
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title(f'Function on {self.domain}')
        if self.name:
            plt.legend()
        plt.grid(True, alpha=0.3)

    def __add__(self, other):
        """Addition of L² functions or with a scalar."""
        if isinstance(other, L2Function):
            if other.space != self.space:
                raise ValueError("Functions must be in the same space")

            if (self.coefficients is not None and
                    other.coefficients is not None):
                new_coeffs = self.coefficients + other.coefficients
                return self.__class__(
                    self.space, coefficients=new_coeffs
                )
            else:
                # For callable functions, create a new callable
                def new_callable(x):
                    return self.evaluate(x) + other.evaluate(x)
                return self.__class__(
                    self.space, evaluate_callable=new_callable
                )
        elif isinstance(other, numbers.Number):
            # Adding a constant
            if self.coefficients is not None:
                new_coeffs = self.coefficients.copy()
                new_coeffs[0] += other  # Assume first coefficient is constant
                return self.__class__(
                    self.space, coefficients=new_coeffs
                )
            else:
                def new_callable(x):
                    return self.evaluate(x) + other
                return self.__class__(
                    self.space, evaluate_callable=new_callable
                )
        else:
            raise TypeError(
                f"Can only add L2Function or scalar "
                f"(int, float, numbers.Number), not {type(other)}"
            )

    def __radd__(self, other):
        """Right addition to support scalar + L2Function."""
        return self.__add__(other)

    def __mul__(self, other):
        """Multiplication of L² functions or by scalars."""
        if isinstance(other, (int, float)):
            # Scalar multiplication
            if self.coefficients is not None:
                return self.__class__(
                    self.space,
                    coefficients=other * self.coefficients
                )
            else:
                def new_callable(x):
                    return other * self.evaluate(x)
                return self.__class__(
                    self.space,
                    evaluate_callable=new_callable
                )
        elif isinstance(other, L2Function):
            # Function multiplication: always use pointwise multiplication
            if self.space != other.space:
                raise ValueError("Functions must be in the same space for "
                                 "multiplication")

            def new_callable(x):
                return self.evaluate(x) * other.evaluate(x)
            return self.__class__(self.space, evaluate_callable=new_callable)
        else:
            raise TypeError(
                f"Cannot multiply L2Function with {type(other)}"
            )

    def __rmul__(self, other):
        """Right multiplication (for scalar * function)."""
        return self.__mul__(other)

    def __repr__(self) -> str:
        return (f"L2Function(domain={self.domain}, "
                f"space_type={self.space_type}, name={self.name})")
