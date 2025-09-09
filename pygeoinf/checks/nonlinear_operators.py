"""
Provides a self-checking mechanism for NonLinearOperator implementations.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..linear_operators import LinearOperator


class NonLinearOperatorAxiomChecks:
    """A mixin for checking the properties of a NonLinearOperator."""

    def _check_derivative_finite_difference(self, x, v, h=1e-7):
        """
        Verifies the derivative using the finite difference formula:
        D[F](x) @ v  ≈  (F(x + h*v) - F(x)) / h
        """
        from ..linear_operators import LinearOperator

        derivative_op = self.derivative(x)

        # 1. Check that the derivative is a valid LinearOperator
        if not isinstance(derivative_op, LinearOperator):
            raise AssertionError("The derivative must be a valid LinearOperator.")
        if not (
            derivative_op.domain == self.domain
            and derivative_op.codomain == self.codomain
        ):
            raise AssertionError("The derivative has a mismatched domain or codomain.")

        # 2. Calculate the analytical derivative's action on a random vector v
        analytic_result = derivative_op(v)

        # 3. Calculate the numerical approximation using the finite difference formula
        x_plus_hv = self.domain.add(x, self.domain.multiply(h, v))
        fx_plus_hv = self(x_plus_hv)
        fx = self(x)
        finite_diff_result = self.codomain.multiply(
            1 / h, self.codomain.subtract(fx_plus_hv, fx)
        )

        # 4. Compare the analytical and numerical results
        diff_norm = self.codomain.norm(
            self.codomain.subtract(analytic_result, finite_diff_result)
        )
        analytic_norm = self.codomain.norm(analytic_result)
        relative_error = diff_norm / (analytic_norm + 1e-12)

        if relative_error > 1e-4:
            raise AssertionError(
                f"Finite difference check failed. Relative error: {relative_error:.2e}"
            )

    def check(self, n_checks: int = 5) -> None:
        """
        Runs randomized checks to validate the operator's derivative.

        Args:
            n_checks: The number of randomized trials to perform.

        Raises:
            AssertionError: If the finite difference check fails.
        """
        print(
            f"\nRunning {n_checks} randomized checks for {self.__class__.__name__}..."
        )
        for _ in range(n_checks):
            x = self.domain.random()
            v = self.domain.random()
            # Ensure the direction vector 'v' is not a zero vector
            if self.domain.norm(v) < 1e-12:
                v = self.domain.random()
            self._check_derivative_finite_difference(x, v)
        print(f"✅ All {n_checks} non-linear operator checks passed successfully.")
