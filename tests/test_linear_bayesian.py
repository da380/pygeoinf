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
from pygeoinf.subspaces import AffineSubspace
from pygeoinf.linear_bayesian import (
    LinearBayesianInversion,
    ConstrainedLinearBayesianInversion,
)

# =============================================================================
# Fixtures for the General Test Problem (5D -> 3D)
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
# Fixtures for Specific Constrained Tests (3D Identity)
# =============================================================================


@pytest.fixture
def r3() -> EuclideanSpace:
    """Provides a 3D Euclidean space."""
    return EuclideanSpace(3)


@pytest.fixture
def identity_problem(r3) -> LinearForwardProblem:
    """
    Forward problem: d = u + e.
    Data Error: sigma = 0.1.
    """
    fp = LinearForwardProblem(
        r3.identity_operator(),
        data_error_measure=GaussianMeasure.from_standard_deviation(r3, 0.1),
    )
    return fp


@pytest.fixture
def standard_prior(r3) -> GaussianMeasure:
    """Prior: u ~ N(0, I)."""
    return GaussianMeasure.from_standard_deviation(r3, 1.0)


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
# Constrained Bayesian Tests
# =============================================================================


class TestConstrainedLinearBayesianInversion:

    def test_prior_conditioning_on_affine_plane(self):
        """
        Test conditioning a Standard Normal prior on the plane z = 1.
        Constraint B(u) = u_z = 1.
        Property Space (codomain of B) is 1D.
        """
        r3 = EuclideanSpace(3)
        standard_prior = GaussianMeasure.from_standard_deviation(r3, 1.0)

        # Constraint: u_z = 1
        codomain = EuclideanSpace(1)
        e_z = r3.basis_vector(2)

        def mapping(u):
            return np.array([r3.inner_product(u, e_z)])

        def adjoint(w):
            return r3.multiply(w[0], e_z)

        B = LinearOperator(r3, codomain, mapping, adjoint_mapping=adjoint)
        w = codomain.from_components(np.array([1.0]))
        constraint = AffineSubspace.from_linear_equation(B, w)

        # We need a dummy forward problem to init the class
        fp = LinearForwardProblem(r3.identity_operator())

        solver = ConstrainedLinearBayesianInversion(fp, standard_prior, constraint)

        # Conditioning involves inverting B C B* (1x1 matrix)
        cond_prior = solver.conditioned_prior_measure(
            solver=CholeskySolver(galerkin=True)
        )

        # Mean should be [0, 0, 1]
        mean = r3.to_components(cond_prior.expectation)
        assert np.allclose(mean, [0.0, 0.0, 1.0])

        # Covariance should be diag(1, 1, 0)
        cov = cond_prior.covariance.matrix(dense=True)
        assert np.allclose(cov, np.diag([1.0, 1.0, 0.0]), atol=1e-10)

    def test_posterior_update_with_constraint(self):
        """
        Test full inversion.
        Data Space is R3. Constraint Space is R1.
        This confirms we can handle different solvers/preconditioners.
        """
        r3 = EuclideanSpace(3)
        # Forward: Identity, Data Error = 0.1
        fp = LinearForwardProblem(
            r3.identity_operator(),
            data_error_measure=GaussianMeasure.from_standard_deviation(r3, 0.1),
        )
        prior = GaussianMeasure.from_standard_deviation(r3, 1.0)

        # Constraint: u_z = 1
        codomain = EuclideanSpace(1)
        e_z = r3.basis_vector(2)

        def mapping(u):
            return np.array([r3.inner_product(u, e_z)])

        def adjoint(w):
            return r3.multiply(w[0], e_z)

        B = LinearOperator(r3, codomain, mapping, adjoint_mapping=adjoint)
        w = codomain.from_components(np.array([1.0]))
        constraint = AffineSubspace.from_linear_equation(B, w)

        solver = ConstrainedLinearBayesianInversion(fp, prior, constraint)

        data = r3.from_components(np.array([10.0, 10.0, 10.0]))

        # We pass different solvers (though same type) to verify the API
        posterior = solver.model_posterior_measure(
            data,
            solver=CholeskySolver(galerkin=True),  # Acts on Data Space (3D)
            constraint_solver=CholeskySolver(
                galerkin=True
            ),  # Acts on Property Space (1D)
        )

        post_mean = r3.to_components(posterior.expectation)

        # Z should be exactly 1.0
        assert np.isclose(post_mean[2], 1.0)
        # X/Y should be approx 9.9
        assert np.allclose(post_mean[:2], 9.9, atol=0.1)
