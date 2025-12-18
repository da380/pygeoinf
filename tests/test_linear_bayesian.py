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


# =============================================================================
# New Sampling Tests for LinearBayesianInversion
# =============================================================================


class TestBayesianSampling:
    """Tests for the sampling capabilities of the posterior measure."""

    def test_posterior_sampling_statistics(
        self,
        forward_problem: LinearForwardProblem,
        model_prior_measure: GaussianMeasure,
        data: np.ndarray,
    ):
        """
        Verifies that the sampling method correctly reproduces the posterior
        mean and covariance. This confirms the 'Perturbed Observation' method works.
        """
        solver = CholeskySolver(galerkin=True)
        inversion = LinearBayesianInversion(forward_problem, model_prior_measure)
        posterior = inversion.model_posterior_measure(data, solver)

        # 1. Check sampler exists
        assert posterior.sample_set, "Posterior should support sampling."

        # 2. Draw a large ensemble of samples
        n_samples = 5000
        samples = posterior.samples(n_samples)

        # 3. Compute sample statistics using the library's built-in helper
        # Note: We use the domain's sample_expectation helper if available,
        # or manual numpy calc for Euclidean spaces.
        space = forward_problem.model_space
        sample_matrix = np.column_stack([space.to_components(s) for s in samples])

        sample_mean = np.mean(sample_matrix, axis=1)
        sample_cov = np.cov(sample_matrix)

        # 4. Compare against analytical posterior
        true_mean = space.to_components(posterior.expectation)
        true_cov = posterior.covariance.matrix(dense=True)

        # Allow for statistical noise (Standard Error ~ 1/sqrt(N))
        # With N=5000, errors should be roughly < 5-10% of variance magnitude
        assert np.allclose(sample_mean, true_mean, atol=0.15)
        assert np.allclose(sample_cov, true_cov, atol=0.15)


# =============================================================================
# New Geometric vs Bayesian Constraint Tests
# =============================================================================


class TestGeometricVsBayesianConstraints:
    """
    Tests the difference between Geometric (Orthogonal) and Bayesian (Oblique)
    conditioning.
    """

    def test_geometric_ignores_prior_covariance(self):
        """
        Scenario:
        - 2D Space.
        - Prior is highly correlated (stretched along y=x).
        - Constraint is u_y = -u_x (orthogonal to the correlation).

        Geometric Projection should snap to the line using the shortest path (Euclidean).
        Bayesian Update should slide along the correlation direction (Mahalanobis).
        """
        r2 = EuclideanSpace(2)

        # 1. Create a correlated prior: Cov = [[1, 0.9], [0.9, 1]]
        # This cloud is stretched along y=x.
        L = np.array([[1.0, 0.0], [0.9, 0.4359]])  # Cholesky factor approx
        cov_matrix = L @ L.T
        prior = GaussianMeasure.from_covariance_matrix(r2, cov_matrix)

        # 2. Define Constraint: u_x + u_y = 2  =>  [1, 1] . u = 2
        # This line is perpendicular to the prior's main correlation axis.
        codomain = EuclideanSpace(1)
        vec_11 = r2.from_components(np.array([1.0, 1.0]))

        def mapping(u):
            return np.array([r2.inner_product(u, vec_11)])

        def adjoint(w):
            return r2.multiply(w[0], vec_11)

        B = LinearOperator(r2, codomain, mapping, adjoint_mapping=adjoint)
        w = codomain.from_components(np.array([2.0]))
        constraint = AffineSubspace.from_linear_equation(B, w)

        # Dummy forward problem
        fp = LinearForwardProblem(r2.identity_operator())

        # 3. Run Geometric (Orthogonal) Inversion
        geo_inv = ConstrainedLinearBayesianInversion(
            fp, prior, constraint, geometric=True
        )
        geo_prior = geo_inv.conditioned_prior_measure(
            solver=CholeskySolver(galerkin=True)
        )
        geo_mean = r2.to_components(geo_prior.expectation)

        # Geometric Logic: Shortest path from (0,0) to x+y=2 is (1,1).
        # It ignores the fact that the prior says (1,1) is "far away" probabilistically.
        assert np.allclose(geo_mean, [1.0, 1.0])

        # 4. Run Bayesian (Statistical) Inversion
        bayes_inv = ConstrainedLinearBayesianInversion(
            fp, prior, constraint, geometric=False
        )
        bayes_prior = bayes_inv.conditioned_prior_measure(
            solver=CholeskySolver(galerkin=True)
        )
        bayes_mean = r2.to_components(bayes_prior.expectation)

        # Bayesian Logic: The prior is correlated (y ~ x).
        # To satisfy x+y=2, it prefers points where x and y are both positive and large
        # because that's where the probability mass is.
        # It should be exactly [1, 1] ONLY if the correlation aligns or is isotropic.
        # Actually, for x+y=const on a symmetric covariance, the mean update
        # is symmetric. Let's check a non-symmetric case or just verify they are identical
        # here because the constraint is symmetric to the covariance.

        # Let's try a case where they MUST differ:
        # Prior Mean = (0,0). Constraint u_x = 1.
        # Covariance strongly correlated: u_y ~ 0.9 u_x.

        # RE-SETUP for clear difference:
        # Constraint: u_x = 1
        e_x = r2.basis_vector(0)

        def map_x(u):
            return np.array([r2.inner_product(u, e_x)])

        def adj_x(w):
            return r2.multiply(w[0], e_x)

        B_x = LinearOperator(r2, codomain, map_x, adjoint_mapping=adj_x)
        w_x = codomain.from_components(np.array([1.0]))
        constraint_x = AffineSubspace.from_linear_equation(B_x, w_x)

        # Geometric: Proj(0,0) onto x=1 is (1,0).
        # It doesn't change y because the constraint is orthogonal to y.
        geo_inv_x = ConstrainedLinearBayesianInversion(
            fp, prior, constraint_x, geometric=True
        )
        geo_mean_x = r2.to_components(
            geo_inv_x.conditioned_prior_measure(
                solver=CholeskySolver(galerkin=True)
            ).expectation
        )
        assert np.allclose(geo_mean_x, [1.0, 0.0])

        # Bayesian: Given u_x=1, we expect u_y to be ~0.9 (correlation).
        bayes_inv_x = ConstrainedLinearBayesianInversion(
            fp, prior, constraint_x, geometric=False
        )
        bayes_mean_x = r2.to_components(
            bayes_inv_x.conditioned_prior_measure(
                solver=CholeskySolver(galerkin=True)
            ).expectation
        )

        # This confirms Bayesian update "warps" the unconstrained variable u_y
        assert np.isclose(bayes_mean_x[0], 1.0)
        assert np.isclose(bayes_mean_x[1], 0.9, atol=0.05)

        # Final check: The two methods produced different results
        assert not np.allclose(geo_mean_x, bayes_mean_x)
