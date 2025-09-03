"""
Concrete tests for specific operator implementations.
"""

import pytest
import numpy as np

from pygeoinf.hilbert_space import EuclideanSpace, Vector
from pygeoinf.nonlinear_operators import NonLinearOperator
from pygeoinf.linear_operators import DiagonalLinearOperator
from .checks.nonlinear_operators import NonLinearOperatorChecks


# Define a simple quadratic operator F(x) = x^2 (element-wise) for testing.
# Its derivative is F'(x) = 2*diag(x), a diagonal linear operator.


class TestElementwiseQuadraticOperator(NonLinearOperatorChecks):
    """
    A concrete test suite for an element-wise quadratic operator.
    It inherits all the basic algebraic checks from NonLinearOperatorChecks.
    """

    # --- Fixture Definition ---

    @pytest.fixture
    def operator(self) -> NonLinearOperator:
        """Provides an instance of F(x) = x^2."""
        space = EuclideanSpace(10)

        # Define the derivative: a diagonal operator with 2*x on the diagonal
        def derivative_func(x: Vector) -> DiagonalLinearOperator:
            return DiagonalLinearOperator(space, space, 2 * x)

        return NonLinearOperator(
            space,
            space,
            lambda x: x**2,  # In Euclidean space, vectors are NumPy arrays
            derivative=derivative_func,
        )

    @pytest.fixture
    def operator2(self, operator: NonLinearOperator) -> NonLinearOperator:
        """Provides a second operator instance for binary operation tests."""
        # For simplicity, we can just return a copy or a slightly modified version.
        return operator  # Or create another one if needed.

    # --- Specific Tests for this Operator ---

    def test_derivative_taylor_approximation(
        self, operator: "NonLinearOperator", x: Vector
    ):
        """
        Verifies the derivative for this specific, well-behaved operator.

        Since we control the operator, we can confidently test its derivative
        using a Taylor approximation.
        """
        h = 1e-6 * np.random.randn(operator.domain.dim)  # small perturbation

        # 1. Calculate the "actual" change in the function's output
        Fx = operator(x)
        Fx_plus_h = operator(x + h)
        actual_change = Fx_plus_h - Fx

        # 2. Calculate the "predicted" change using the linear derivative
        derivative_at_x = operator.derivative(x)
        predicted_change = derivative_at_x(h)

        # 3. Assert that the actual and predicted changes are very close
        assert np.allclose(actual_change, predicted_change, rtol=1e-5)
