"""
Provides a self-checking mechanism for NonLinearOperator implementations.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    pass


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

    def _check_add_derivative(self, op1, op2, x, v):
        """Verifies the sum rule for derivatives: (F+G)' = F' + G'"""
        if not (op1.has_derivative and op2.has_derivative):
            return  # Skip if derivatives aren't defined

        # Derivative of the sum of operators
        sum_op = op1 + op2
        derivative_of_sum = sum_op.derivative(x)

        # Sum of the individual derivatives
        sum_of_derivatives = op1.derivative(x) + op2.derivative(x)

        # Compare their action on a random vector
        res1 = derivative_of_sum(v)
        res2 = sum_of_derivatives(v)

        diff_norm = self.codomain.norm(self.codomain.subtract(res1, res2))
        if diff_norm > 1e-9:
            raise AssertionError("Axiom failed: Derivative of sum is incorrect.")

    def _check_scalar_mul_derivative(self, op, x, v, a):
        """Verifies the scalar multiple rule: (a*F)' = a*F'"""
        if not op.has_derivative:
            return

        # Derivative of the scaled operator
        scaled_op = a * op
        derivative_of_scaled = scaled_op.derivative(x)

        # Scaled original derivative
        scaled_derivative = a * op.derivative(x)

        # Compare their action
        res1 = derivative_of_scaled(v)
        res2 = scaled_derivative(v)

        diff_norm = self.codomain.norm(self.codomain.subtract(res1, res2))
        if diff_norm > 1e-9:
            raise AssertionError(
                "Axiom failed: Derivative of scalar multiple is incorrect."
            )

    def _check_matmul_derivative(self, op1, op2, x, v):
        """Verifies the chain rule for derivatives: (F o G)'(x) = F'(G(x)) @ G'(x)"""
        if not (op1.has_derivative and op2.has_derivative):
            return
        if op1.domain != op2.codomain:
            return  # Skip if not composable

        # Derivative of the composed operator
        composed_op = op1 @ op2
        derivative_of_composed = composed_op.derivative(x)

        # Apply the chain rule manually
        gx = op2(x)
        chain_rule_derivative = op1.derivative(gx) @ op2.derivative(x)

        # Compare their action
        res1 = derivative_of_composed(v)
        res2 = chain_rule_derivative(v)

        diff_norm = op1.codomain.norm(op1.codomain.subtract(res1, res2))
        if diff_norm > 1e-9:
            raise AssertionError(
                "Axiom failed: Chain rule for derivatives is incorrect."
            )

    def check(self, n_checks: int = 5, op2=None) -> None:
        """
        Runs randomized checks to validate the operator's derivative and
        its algebraic properties.

        Args:
            n_checks: The number of randomized trials to perform.
            op2: An optional second operator for testing algebraic rules.
        """
        print(
            f"\nRunning {n_checks} randomized checks for {self.__class__.__name__}..."
        )
        for _ in range(n_checks):
            x = self.domain.random()
            v = self.domain.random()
            a = np.random.randn()

            # Ensure the direction vector 'v' is not a zero vector
            if self.domain.norm(v) < 1e-12:
                v = self.domain.random()

            # Original check
            self._check_derivative_finite_difference(x, v)

            # New algebraic checks
            self._check_scalar_mul_derivative(self, x, v, a)
            if op2:
                self._check_add_derivative(self, op2, x, v)
                self._check_matmul_derivative(self, op2, x, v)

        print(f"✅ All {n_checks} non-linear operator checks passed successfully.")
