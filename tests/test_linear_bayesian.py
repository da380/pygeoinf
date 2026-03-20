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
    model_space = EuclideanSpace(dim=50)
    data_space = EuclideanSpace(dim=30)
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

        # API CHANGE: Solver is now passed to the subspace factory
        constraint = AffineSubspace.from_linear_equation(
            B, w, CholeskySolver(galerkin=True)
        )

        fp = LinearForwardProblem(r3.identity_operator())
        solver = ConstrainedLinearBayesianInversion(fp, standard_prior, constraint)

        # API CHANGE: conditioned_prior_measure no longer takes a solver argument
        cond_prior = solver.conditioned_prior_measure()

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
        This confirms we can handle separate solvers for data and constraints.
        """
        r3 = EuclideanSpace(3)
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

        # 1. Define Constraint (Solver for Property Space attached here)
        constraint = AffineSubspace.from_linear_equation(
            B, w, CholeskySolver(galerkin=True)
        )

        solver = ConstrainedLinearBayesianInversion(fp, prior, constraint)

        data = r3.from_components(np.array([10.0, 10.0, 10.0]))

        # 2. Solve (Solver for Data Space passed here)
        # API CHANGE: constraint_solver argument removed.
        posterior = solver.model_posterior_measure(
            data,
            CholeskySolver(galerkin=True),
        )

        post_mean = r3.to_components(posterior.expectation)

        # Z should be exactly 1.0 (hard constraint)
        assert np.isclose(post_mean[2], 1.0)
        # X/Y should be approx 9.9 (Bayesian update)
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
        mean and covariance.
        """
        solver = CholeskySolver(galerkin=True)
        inversion = LinearBayesianInversion(forward_problem, model_prior_measure)
        posterior = inversion.model_posterior_measure(data, solver)

        # 1. Check sampler exists
        assert posterior.sample_set, "Posterior should support sampling."

        # 2. Draw samples
        n_samples = 5000
        samples = posterior.samples(n_samples)

        # 3. Compute statistics
        space = forward_problem.model_space
        sample_matrix = np.column_stack([space.to_components(s) for s in samples])

        sample_mean = np.mean(sample_matrix, axis=1)
        sample_cov = np.cov(sample_matrix)

        # 4. Compare
        true_mean = space.to_components(posterior.expectation)
        true_cov = posterior.covariance.matrix(dense=True)

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
        Scenario: Prior is correlated (y~x). Constraint is y=-x.
        """
        r2 = EuclideanSpace(2)

        # 1. Correlated prior
        L = np.array([[1.0, 0.0], [0.9, 0.4359]])
        cov_matrix = L @ L.T
        prior = GaussianMeasure.from_covariance_matrix(r2, cov_matrix)

        # 2. Constraint: u_x + u_y = 2
        codomain = EuclideanSpace(1)
        vec_11 = r2.from_components(np.array([1.0, 1.0]))

        def mapping(u):
            return np.array([r2.inner_product(u, vec_11)])

        def adjoint(w):
            return r2.multiply(w[0], vec_11)

        B = LinearOperator(r2, codomain, mapping, adjoint_mapping=adjoint)
        w = codomain.from_components(np.array([2.0]))

        # Attach solver to subspace definition
        constraint = AffineSubspace.from_linear_equation(
            B, w, CholeskySolver(galerkin=True)
        )

        fp = LinearForwardProblem(r2.identity_operator())

        # 3. Geometric Inversion
        geo_inv = ConstrainedLinearBayesianInversion(
            fp, prior, constraint, geometric=True
        )
        # API CHANGE: No solver passed here
        geo_prior = geo_inv.conditioned_prior_measure()
        geo_mean = r2.to_components(geo_prior.expectation)

        # Geometric: Shortest path from (0,0) to x+y=2 is (1,1).
        assert np.allclose(geo_mean, [1.0, 1.0])

        # 4. Second case: Constraint u_x = 1 (to force difference)
        e_x = r2.basis_vector(0)

        def map_x(u):
            return np.array([r2.inner_product(u, e_x)])

        def adj_x(w):
            return r2.multiply(w[0], e_x)

        B_x = LinearOperator(r2, codomain, map_x, adjoint_mapping=adj_x)
        w_x = codomain.from_components(np.array([1.0]))

        constraint_x = AffineSubspace.from_linear_equation(
            B_x, w_x, CholeskySolver(galerkin=True)
        )

        # Geometric
        geo_inv_x = ConstrainedLinearBayesianInversion(
            fp, prior, constraint_x, geometric=True
        )
        geo_mean_x = r2.to_components(geo_inv_x.conditioned_prior_measure().expectation)
        assert np.allclose(geo_mean_x, [1.0, 0.0])

        # Bayesian
        bayes_inv_x = ConstrainedLinearBayesianInversion(
            fp, prior, constraint_x, geometric=False
        )
        bayes_mean_x = r2.to_components(
            bayes_inv_x.conditioned_prior_measure().expectation
        )

        # Bayesian respects correlation (y ~ 0.9x)
        assert np.isclose(bayes_mean_x[0], 1.0)
        assert np.isclose(bayes_mean_x[1], 0.9, atol=0.05)


# =============================================================================
# New Tests for Diagonal Preconditioner
# =============================================================================


class TestDiagonalNormalPreconditioner:
    """Tests for the diagonal normal preconditioner optimization."""

    def test_exact_diagonal_no_blocks(
        self,
        forward_problem: LinearForwardProblem,
        model_prior_measure: GaussianMeasure,
    ):
        """
        Verifies that the matrix-free diagonal computation exactly matches
        the diagonal of the fully assembled dense normal matrix.
        """
        inversion = LinearBayesianInversion(forward_problem, model_prior_measure)

        # 1. Compute using the new optimized method
        preconditioner = inversion.diagonal_normal_preconditioner()

        # The preconditioner is the *inverse* of the diagonal normal operator,
        # so its diagonal entries should be 1 / diag(N)
        precon_diagonal = preconditioner.extract_diagonal()
        approx_normal_diagonal = 1.0 / precon_diagonal

        # 2. Get dense matrices for exact brute-force comparison
        A = forward_problem.forward_operator.matrix(dense=True)
        Q = model_prior_measure.covariance.matrix(dense=True)
        R = forward_problem.data_error_measure.covariance.matrix(dense=True)

        # N = A Q A^T + R
        exact_normal_matrix = A @ Q @ A.T + R
        exact_diagonal = np.diag(exact_normal_matrix)

        # 3. Compare
        assert np.allclose(approx_normal_diagonal, exact_diagonal)

    def test_block_averaging(
        self,
        forward_problem: LinearForwardProblem,
        model_prior_measure: GaussianMeasure,
    ):
        """
        Verifies that providing blocks correctly averages the basis vectors
        to compute a representative regional variance.
        """
        inversion = LinearBayesianInversion(forward_problem, model_prior_measure)

        data_dim = forward_problem.data_space.dim

        # We block indices [0, 1] together, leave [2] alone, and lump the rest
        # into a third block so the entire data space is perfectly partitioned.
        blocks = [[0, 1], [2], list(range(3, data_dim))]

        preconditioner = inversion.diagonal_normal_preconditioner(blocks=blocks)

        # Extract diagonal to verify values
        approx_normal_diagonal = preconditioner.inverse.extract_diagonal()

        model_space = forward_problem.model_space
        data_space = forward_problem.data_space
        A_adj = forward_problem.forward_operator.adjoint
        Q_op = model_prior_measure.covariance

        # 1. Manual check for block [0, 1]
        v01 = data_space.from_components(np.array([0.5, 0.5] + [0.0] * (data_dim - 2)))
        f_v01 = A_adj(v01)
        aqa_01 = model_space.inner_product(f_v01, Q_op(f_v01))

        # R is identity in the fixture, so R_00 = R_11 = 1.0
        expected_0 = aqa_01 + 1.0
        expected_1 = aqa_01 + 1.0

        assert np.isclose(approx_normal_diagonal[0], expected_0)
        assert np.isclose(approx_normal_diagonal[1], expected_1)

        # 2. Manual check for block [2] (just index 2)
        v2 = data_space.from_components(
            np.array([0.0, 0.0, 1.0] + [0.0] * (data_dim - 3))
        )
        f_v2 = A_adj(v2)
        aqa_2 = model_space.inner_product(f_v2, Q_op(f_v2))
        expected_2 = aqa_2 + 1.0

        assert np.isclose(approx_normal_diagonal[2], expected_2)

    def test_block_validation(
        self,
        forward_problem: LinearForwardProblem,
        model_prior_measure: GaussianMeasure,
    ):
        """
        Verifies that malformed blocks raise the appropriate errors.
        """
        inversion = LinearBayesianInversion(forward_problem, model_prior_measure)
        data_dim = forward_problem.data_space.dim

        # Missing index 2
        missing_blocks = [[0, 1]] + [list(range(3, data_dim))]
        with pytest.raises(ValueError, match="must exactly partition"):
            inversion.diagonal_normal_preconditioner(blocks=missing_blocks)

        # Duplicate index 1
        duplicate_blocks = [[0, 1], [1, 2]] + [list(range(3, data_dim))]
        with pytest.raises(ValueError, match="must exactly partition"):
            inversion.diagonal_normal_preconditioner(blocks=duplicate_blocks)

        # Out of bounds index (swapping index 2 for data_dim)
        out_of_bounds_blocks = [[0, 1], [data_dim]] + [list(range(3, data_dim))]
        with pytest.raises(ValueError, match="out of bounds"):
            inversion.diagonal_normal_preconditioner(blocks=out_of_bounds_blocks)


# =============================================================================
# New Tests for Sparse Localized Preconditioner
# =============================================================================


class TestSparseLocalizedPreconditioner:
    """Tests for the sparse localized normal preconditioner."""

    def test_exact_dense_match(
        self,
        forward_problem: LinearForwardProblem,
        model_prior_measure: GaussianMeasure,
    ):
        """
        Verifies that with a single global block and sufficient rank, the
        preconditioner exactly matches the inverse of the dense normal matrix.
        """
        inversion = LinearBayesianInversion(forward_problem, model_prior_measure)

        # 1. One block containing all data indices
        data_dim = forward_problem.data_space.dim
        blocks = [list(range(data_dim))]

        # 2. Build preconditioner with rank >= data_dim to ensure exact reconstruction
        preconditioner = inversion.sparse_localized_preconditioner(
            blocks, rank=data_dim
        )

        # 3. Get exact dense matrices for comparison
        A = forward_problem.forward_operator.matrix(dense=True)
        Q = model_prior_measure.covariance.matrix(dense=True)
        R = forward_problem.data_error_measure.covariance.matrix(dense=True)

        # N = A Q A^T + R
        exact_normal_matrix = A @ Q @ A.T + R
        exact_inverse_matrix = np.linalg.inv(exact_normal_matrix)

        # 4. Extract the dense matrix from our pygeoinf LinearOperator
        approx_inverse_matrix = preconditioner.matrix(dense=True)

        # 5. Compare (should be identical up to floating point precision)
        assert np.allclose(approx_inverse_matrix, exact_inverse_matrix)

    def test_overlapping_blocks(
        self,
        forward_problem: LinearForwardProblem,
        model_prior_measure: GaussianMeasure,
        data: np.ndarray,
    ):
        """
        Verifies that the COO sparse assembly correctly handles overlapping
        sub-blocks spanning the entire data space without throwing symmetry errors.
        """
        inversion = LinearBayesianInversion(forward_problem, model_prior_measure)

        # 1. Define overlapping blocks to tile the new 30-dimensional data space
        data_dim = forward_problem.data_space.dim

        # Creates a chain of overlapping blocks: [0...14], [10...24], [20...29]
        blocks = [list(range(0, 15)), list(range(10, 25)), list(range(20, data_dim))]

        # 2. Build preconditioner (should sum overlaps implicitly)
        # Using rank=10 forces a true low-rank approximation of the size-15 blocks
        preconditioner = inversion.sparse_localized_preconditioner(blocks, rank=10)

        # 3. Apply it to a random data vector
        result = preconditioner(data)

        # 4. Verify the output is a valid element of the data space
        assert forward_problem.data_space.is_element(result)
