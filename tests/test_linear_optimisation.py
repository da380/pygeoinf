"""
Tests for the linear_optimisation module.
"""

import pytest
import numpy as np
from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.linear_operators import LinearOperator
from pygeoinf.gaussian_measure import GaussianMeasure
from pygeoinf.forward_problem import LinearForwardProblem
from pygeoinf.linear_solvers import CholeskySolver
from pygeoinf.linear_optimisation import (
    LinearLeastSquaresInversion,
    LinearMinimumNormInversion,
)

# =============================================================================
# Fixtures for the Test Problem
# =============================================================================


@pytest.fixture
def forward_problem() -> LinearForwardProblem:
    """
    Provides a simple, underdetermined forward problem, which is typical
    for inverse problems.
    """
    model_space = EuclideanSpace(dim=5)
    data_space = EuclideanSpace(dim=3)
    matrix = np.random.randn(data_space.dim, model_space.dim)
    forward_operator = LinearOperator.from_matrix(model_space, data_space, matrix)
    error_measure = GaussianMeasure.from_standard_deviation(data_space, 1.0)
    return LinearForwardProblem(forward_operator, data_error_measure=error_measure)


@pytest.fixture
def true_model(forward_problem: LinearForwardProblem) -> np.ndarray:
    """Provides a random 'true' model from the model space."""
    return forward_problem.model_space.random()


@pytest.fixture
def synthetic_data(
    forward_problem: LinearForwardProblem, true_model: np.ndarray
) -> np.ndarray:
    """Generates synthetic noisy data from the true model."""
    return forward_problem.synthetic_data(true_model)


@pytest.fixture
def small_norm_data(forward_problem: LinearForwardProblem) -> np.ndarray:
    """
    Provides a small-norm data vector to test the case where the zero model
    is a valid solution.
    """
    return forward_problem.data_space.random() * 0.1


# =============================================================================
# Tests for LinearLeastSquaresInversion
# =============================================================================


class TestLinearLeastSquaresInversion:
    """A suite of tests for the LinearLeastSquaresInversion class."""

    def test_least_squares_analytical_solution(
        self, forward_problem: LinearForwardProblem
    ):
        """
        Tests that the least-squares solution matches the analytical solution
        for an underdetermined problem.
        """
        damping = 0.1
        solver = CholeskySolver(galerkin=True)
        data = forward_problem.data_space.random()

        # 1. Compute the solution using the library's inversion class
        lsq_inversion = LinearLeastSquaresInversion(forward_problem)
        lsq_operator = lsq_inversion.least_squares_operator(damping, solver)
        model_solution = lsq_operator(data)

        # 2. Compute the solution analytically for comparison
        # The solution is u = (A^T C_e^-1 A + alpha*I)^-1 A^T C_e^-1 d
        A = forward_problem.forward_operator.matrix(dense=True)
        Ce_inv = forward_problem.data_error_measure.inverse_covariance.matrix(
            dense=True
        )

        normal_matrix = A.T @ Ce_inv @ A + damping * np.eye(A.shape[1])
        rhs = A.T @ Ce_inv @ data

        expected_solution = np.linalg.solve(normal_matrix, rhs)

        # 3. Compare the results
        assert np.allclose(model_solution, expected_solution)


# =============================================================================
# Tests for LinearMinimumNormInversion
# =============================================================================


class TestLinearMinimumNormInversion:
    """A suite of tests for the LinearMinimumNormInversion class."""

    def test_discrepancy_principle_zero_solution(
        self, forward_problem: LinearForwardProblem, small_norm_data: np.ndarray
    ):
        """
        Tests that the minimum-norm solution correctly returns the zero model
        when it provides a sufficient fit to the data.
        """
        solver = CholeskySolver(galerkin=True)
        significance_level = 0.95
        target_chi_squared = forward_problem.critical_chi_squared(significance_level)
        zero_model = forward_problem.model_space.zero
        chi_squared_zero = forward_problem.chi_squared(zero_model, small_norm_data)

        if chi_squared_zero > target_chi_squared:
            pytest.skip("Random data generated was too large for this test case.")

        min_norm_inversion = LinearMinimumNormInversion(forward_problem)
        min_norm_operator = min_norm_inversion.minimum_norm_operator(
            solver, significance_level=significance_level
        )
        model_solution = min_norm_operator(small_norm_data)

        assert np.allclose(model_solution, zero_model)

    def test_discrepancy_principle_search(
        self, forward_problem: LinearForwardProblem, synthetic_data: np.ndarray
    ):
        """
        Tests that the minimum-norm solution finds a non-zero model that
        satisfies the discrepancy principle when the data misfit is large.
        """
        solver = CholeskySolver(galerkin=True)
        significance_level = 0.95
        target_chi_squared = forward_problem.critical_chi_squared(significance_level)

        min_norm_inversion = LinearMinimumNormInversion(forward_problem)
        min_norm_operator = min_norm_inversion.minimum_norm_operator(
            solver, significance_level=significance_level
        )
        model_solution = min_norm_operator(synthetic_data)

        final_chi_squared = forward_problem.chi_squared(model_solution, synthetic_data)
        assert final_chi_squared <= (target_chi_squared + 1.0e-5)
