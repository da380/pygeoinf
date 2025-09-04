"""
Tests for the non-linear optimisation solvers.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from pygeoinf.hilbert_space import EuclideanSpace, Vector
from pygeoinf.linear_operators import LinearOperator
from pygeoinf.nonlinear_forms import NonLinearForm
from pygeoinf.linear_forms import LinearForm
from pygeoinf.nonlinear_optimisation import ScipyUnconstrainedOptimiser


# =============================================================================
# Test Problems
# =============================================================================


@pytest.fixture
def quadratic_problem() -> dict:
    """
    Provides a simple quadratic optimisation problem with a known solution.

    The functional is f(x) = 0.5 * x.T @ A @ x - b.T @ x.
    The minimum is the solution to the linear system A @ x = b.
    The NonLinearForm is always created with a gradient and a Hessian.
    """
    dim = 10
    space = EuclideanSpace(dim)

    # Create a random positive-definite matrix A
    _A = np.random.randn(dim, dim)
    A = _A.T @ _A + 0.1 * np.eye(dim)
    A_op = LinearOperator.from_matrix(space, space, A, galerkin=True)

    # Create a random vector b
    b = np.random.randn(dim)
    b_form = LinearForm(space, components=b)

    # The known solution is the solution to Ax = b
    known_solution = np.linalg.solve(A, b)

    # Define the quadratic functional
    def mapping(x: Vector) -> float:
        return 0.5 * space.inner_product(x, A_op(x)) - b_form(x)

    def gradient(x: Vector) -> Vector:
        # Gradient is Ax - b
        grad_vec = space.subtract(A_op(x), space.from_dual(b_form))
        return grad_vec

    def hessian(x: Vector) -> LinearOperator:
        # Hessian is the constant operator A
        return A_op

    quadratic_form = NonLinearForm(space, mapping, gradient=gradient, hessian=hessian)

    return {
        "form": quadratic_form,
        "x0": space.zero,
        "solution": known_solution,
    }


# =============================================================================
# Test Suite
# =============================================================================


class TestScipyUnconstrainedOptimiser:
    """
    A test suite for the ScipyUnconstrainedOptimiser class.
    """

    @pytest.mark.parametrize(
        "method",
        [
            "BFGS",
            "Newton-CG",
            "CG",
            "L-BFGS-B",
        ],
    )
    def test_quadratic_minimisation(self, method: str, quadratic_problem: dict):

        # 1. Unpack the test problem
        form = quadratic_problem["form"]
        x0 = quadratic_problem["x0"]
        expected_solution = quadratic_problem["solution"]
        space = form.domain

        # 2. Set up and run the optimiser
        tol = 1e-12
        optimiser = ScipyUnconstrainedOptimiser(method, tol=tol)
        result_vec = optimiser.minimize(form, x0)

        # 3. Check the result
        found_solution = space.to_components(result_vec)
        assert_allclose(found_solution, expected_solution, rtol=1e-5)
