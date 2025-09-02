"""
Tests for the linear_bayesian module.
"""

import pytest
import numpy as np
from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.linear_operators import LinearOperator
from pygeoinf.gaussian_measure import GaussianMeasure
from pygeoinf.forward_problem import LinearForwardProblem
from pygeoinf.linear_solvers import CholeskySolver
from pygeoinf.linear_bayesian import (
    LinearBayesianInversion,
    LinearBayesianInference,
)

# =============================================================================
# Fixtures for the Test Problem
# =============================================================================


@pytest.fixture
def forward_problem() -> LinearForwardProblem:
    """
    Provides a simple, underdetermined forward problem.
    """
    model_space = EuclideanSpace(dim=5)
    data_space = EuclideanSpace(dim=3)
    matrix = np.random.randn(data_space.dim, model_space.dim)
    forward_operator = LinearOperator.from_matrix(model_space, data_space, matrix)
    error_measure = GaussianMeasure.from_standard_deviation(data_space, 1.0)
    return LinearForwardProblem(forward_operator, data_error_measure=error_measure)


@pytest.fixture
def model_prior_measure(forward_problem: LinearForwardProblem) -> GaussianMeasure:
    """Provides a prior measure on the model space."""
    return GaussianMeasure.from_standard_deviation(forward_problem.model_space, 1.0)


@pytest.fixture
def data(forward_problem: LinearForwardProblem) -> np.ndarray:
    """Provides a random data vector."""
    return forward_problem.data_space.random()


# =============================================================================
# Tests for LinearBayesianInversion
# =============================================================================


class TestLinearBayesianInversion:
    """A suite of tests for the LinearBayesianInversion class."""

    def test_posterior_measure(
        self,
        forward_problem: LinearForwardProblem,
        model_prior_measure: GaussianMeasure,
        data: np.ndarray,
    ):
        """
        Tests that the posterior measure matches the analytical solution.
        """
        solver = CholeskySolver(galerkin=True)

        # 1. Compute the posterior using the library's inversion class
        inversion = LinearBayesianInversion(forward_problem, model_prior_measure)
        posterior = inversion.model_posterior_measure(data, solver)

        # 2. Compute the posterior mean and covariance analytically
        A = forward_problem.forward_operator.matrix(dense=True)
        Cu = model_prior_measure.covariance.matrix(dense=True)
        Ce = forward_problem.data_error_measure.covariance.matrix(dense=True)

        # Normal operator C_d = A C_u A^T + C_e
        Cd_inv = np.linalg.inv(A @ Cu @ A.T + Ce)

        # Posterior covariance: C_post = C_u - C_u A^T C_d^-1 A C_u
        expected_cov = Cu - Cu @ A.T @ Cd_inv @ A @ Cu

        # Posterior mean: u_post = u_prior + C_u A^T C_d^-1 (d - A u_prior)
        # (Assuming prior mean is zero)
        expected_mean = Cu @ A.T @ Cd_inv @ data

        # 3. Compare the results
        actual_mean = posterior.expectation
        actual_cov = posterior.covariance.matrix(dense=True)

        assert np.allclose(actual_mean, expected_mean)
        assert np.allclose(actual_cov, expected_cov)


# =============================================================================
# Tests for LinearBayesianInference
# =============================================================================


class TestLinearBayesianInference:
    """A suite of tests for the LinearBayesianInference class."""

    def test_property_posterior(
        self,
        forward_problem: LinearForwardProblem,
        model_prior_measure: GaussianMeasure,
        data: np.ndarray,
    ):
        """
        Tests that the posterior for a derived property is correct.
        """
        solver = CholeskySolver(galerkin=True)
        model_space = forward_problem.model_space

        # 1. Define a simple property operator (e.g., average of model components)
        prop_matrix = np.ones((1, model_space.dim)) / model_space.dim
        property_operator = LinearOperator.from_matrix(
            model_space, EuclideanSpace(1), prop_matrix
        )

        # 2. Compute the property posterior using the library
        inference = LinearBayesianInference(
            forward_problem, model_prior_measure, property_operator
        )
        prop_posterior = inference.property_posterior_measure(data, solver)

        # 3. Compute the property posterior analytically
        # First, get the analytical model posterior
        A = forward_problem.forward_operator.matrix(dense=True)
        Cu = model_prior_measure.covariance.matrix(dense=True)
        Ce = forward_problem.data_error_measure.covariance.matrix(dense=True)
        Cd_inv = np.linalg.inv(A @ Cu @ A.T + Ce)
        model_post_cov = Cu - Cu @ A.T @ Cd_inv @ A @ Cu
        model_post_mean = Cu @ A.T @ Cd_inv @ data

        # Then, transform it using the property operator B
        # E[B*u] = B * E[u]
        # Cov[B*u] = B * Cov[u] * B^T
        B = prop_matrix
        expected_prop_mean = B @ model_post_mean
        expected_prop_cov = B @ model_post_cov @ B.T

        # 4. Compare the results
        actual_prop_mean = prop_posterior.expectation
        actual_prop_cov = prop_posterior.covariance.matrix(dense=True)

        assert np.allclose(actual_prop_mean, expected_prop_mean)
        assert np.allclose(actual_prop_cov, expected_prop_cov)
