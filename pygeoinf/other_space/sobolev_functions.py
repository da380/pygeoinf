"""
Sobolev functions on interval domains.

This module provides function objects that live on IntervalDomain with
Sobolev regularity properties, specialized from the base L2Function class.
"""

import numpy as np
from typing import Union, Optional
from .l2_functions import L2Function


class SobolevFunction(L2Function):
    """
    A function in H^s([a,b]) with Sobolev regularity properties.

    This class represents a function with Sobolev regularity that knows about
    the Sobolev space it belongs to. It extends L2Function with
    Sobolev-specific functionality such as regularity constraints and weak
    derivatives.

    Note: Point evaluation is only well-defined for s > d/2 where s is the
    Sobolev order and d is the spatial dimension. For intervals (d=1),
    point evaluation requires s > 1/2.
    """

    @property
    def space_type(self):
        """Get the type of space this function belongs to."""
        return "Sobolev"

    @property
    def sobolev_order(self) -> float:
        """Get the Sobolev order from the space."""
        return self.space.order

    @property
    def boundary_conditions(self) -> Optional[dict]:
        """Get boundary conditions (if any)."""
        return self.space.boundary_conditions

    def evaluate(self, x: Union[float, np.ndarray],
                 check_domain: bool = True) -> Union[float, np.ndarray]:
        """
        Point evaluation: f(x).

        This is mathematically well-defined for Sobolev functions with s > d/2
        due to the Sobolev embedding theorem. For intervals (d=1), this
        requires s > 1/2, which ensures the function has a continuous
        representative.

        Args:
            x: Point(s) at which to evaluate
            check_domain: Whether to check domain membership

        Returns:
            Function value(s)

        Raises:
            ValueError: If point evaluation is not well-defined for this
                       Sobolev order (s ≤ 1/2 for intervals)
        """
        # Check mathematical validity of point evaluation
        if self.sobolev_order <= 0.5:  # d/2 = 1/2 for intervals
            raise ValueError(
                f"Point evaluation not well-defined for "
                f"H^{self.sobolev_order} on 1D domain. "
                f"Sobolev embedding theorem requires s > 1/2."
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
        # Note: basis functions are themselves SobolevFunctions, so this is
        # recursive but basis functions should have evaluate_callable defined
        basis_evals = np.array([bf.evaluate(x_array, check_domain=False)
                                for bf in basis_functions])
        # basis_evals shape: (n_basis, n_points)
        # Linear combination: sum_k c_k * phi_k(x)
        result = np.tensordot(coeffs, basis_evals, axes=([0], [0]))
        return result[0] if is_scalar else result

    def sobolev_norm(self):
        """Sobolev norm: ||u||_H^s"""
        return np.sqrt(self.space.inner_product(self, self))

    def sobolev_inner_product(self, other):
        """Sobolev inner product with another Sobolev function."""
        if not isinstance(other, SobolevFunction):
            raise TypeError("Can only compute Sobolev inner product with "
                            "another SobolevFunction")
        if self.space != other.space:
            raise ValueError("Functions must be in the same Sobolev space")
        return self.space.inner_product(self, other)

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

        print("Weak derivative is not implemented yet.")
        pass

    def __repr__(self) -> str:
        return (f"SobolevFunction(domain={self.domain}, "
                f"order={self.sobolev_order}, name={self.name})")
