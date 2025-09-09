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
    Provides a simple, deterministic quadratic optimisation problem with a
    known solution.

    The functional is f(x) = sum_{i=0 to n-1} c_i * (x_i - s_i)^2,
    which is a classic convex bowl shape.
    The minimum is at x = s.
    """
    dim = 10
    space = EuclideanSpace(dim)

    # Define the coefficients and the shift for the quadratic bowl
    coeffs = np.arange(1, dim + 1)  # Make it slightly non-uniform
    shifts = np.arange(1, dim + 1)  # The known solution

    # The quadratic form f(x) = 0.5 * x.T@A@x - b.T@x + const
    # corresponds to A = diag(2*coeffs) and b = 2*coeffs*shifts
    A = np.diag(2 * coeffs)
    b = 2 * coeffs * shifts

    A_op = LinearOperator.from_matrix(space, space, A, galerkin=True)
    b_form = LinearForm(space, components=b)

    # The known solution is simply the vector of shifts
    known_solution = shifts

    # Define the quadratic functional
    def mapping(x: Vector) -> float:
        return 0.5 * space.inner_product(x, A_op(x)) - b_form(x)

    def gradient(x: Vector) -> Vector:
        # Gradient is Ax - b
        return space.subtract(A_op(x), space.from_dual(b_form))

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
            pytest.param(
                "Nelder-Mead",
                marks=pytest.mark.xfail(
                    reason="Nelder-Mead is known to fail to converge on this problem"
                ),
            ),
            "Powell",
        ],
    )
    def test_quadratic_minimisation(self, method: str, quadratic_problem: dict):

        # 1. Unpack the test problem
        form = quadratic_problem["form"]
        x0 = quadratic_problem["x0"]
        expected_solution = quadratic_problem["solution"]
        space = form.domain

        # 2. Set different options for different solver types
        if method == "Nelder-Mead":
            # This is expected to fail, but we set the options we tested with.
            options = {"maxiter": 5000, "xatol": 1e-4, "fatol": 1e-6}
            assertion_rtol = 1e-2
            optimiser = ScipyUnconstrainedOptimiser(method, **options)
        elif method == "Powell":
            # Powell works well, but we use a slightly looser tolerance
            # as it's a derivative-free method.
            options = {"maxiter": 5000, "xtol": 1e-5}
            assertion_rtol = 1e-3
            optimiser = ScipyUnconstrainedOptimiser(method, **options)
        else:
            # Gradient-based methods should be very precise.
            optimiser = ScipyUnconstrainedOptimiser(method, tol=1e-12)
            assertion_rtol = 1e-4

        # 3. Run the optimiser
        result_vec = optimiser.minimize(form, x0)

        # 4. Check the result with the appropriate tolerance
        found_solution = space.to_components(result_vec)
        assert_allclose(found_solution, expected_solution, rtol=assertion_rtol)
