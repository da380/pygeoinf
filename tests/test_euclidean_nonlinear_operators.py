"""
Concrete tests for specific non-linear operator implementations.
"""

import pytest
import numpy as np

from pygeoinf.hilbert_space import EuclideanSpace, Vector
from pygeoinf.nonlinear_operators import NonLinearOperator
from pygeoinf.linear_operators import DiagonalLinearOperator


@pytest.fixture
def quadratic_operator() -> NonLinearOperator:
    """Provides an instance of F(x) = x^2 (element-wise)."""
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


def test_elementwise_quadratic_operator_axioms(quadratic_operator: NonLinearOperator):
    """
    Verifies that the operator satisfies the non-linear operator axioms
    by calling its internal self-check method.
    """
    quadratic_operator.check(n_checks=5)


def test_derivative_taylor_approximation(quadratic_operator: NonLinearOperator):
    """
    Verifies the derivative for this specific, well-behaved operator
    using a Taylor approximation.
    """
    operator = quadratic_operator
    x = operator.domain.random()
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
