"""
Tests for the linear_bayesian module.
"""

import pytest
import numpy as np


from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.gaussian_measure import GaussianMeasure
from pygeoinf.forward_problem import LinearForwardProblem
from pygeoinf.linear_solvers import CholeskySolver, LUSolver, ResidualTrackingCallback
from pygeoinf.linear_bayesian import LinearBayesianInversion
from pygeoinf.affine_operators import AffineOperator
from pygeoinf.linear_operators import (
    LinearOperator,
    DenseMatrixLinearOperator,
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

    def test_posterior_expectation_operator(
        self,
        forward_problem: LinearForwardProblem,
        model_prior_measure: GaussianMeasure,
        data: np.ndarray,
    ):
        """
        Tests that the posterior expectation operator correctly maps data to the
        analytical posterior mean, handling both strictly linear (zero-mean)
        and affine (non-zero mean) cases.
        """
        solver = CholeskySolver(galerkin=True)

        # --- Case 1: Zero Mean (Should return a purely LinearOperator) ---
        inversion_zero = LinearBayesianInversion(forward_problem, model_prior_measure)
        post_op_zero = inversion_zero.posterior_expectation_operator(solver)

        assert isinstance(post_op_zero, LinearOperator)
        assert not isinstance(post_op_zero, AffineOperator)

        expected_mean_zero = inversion_zero.model_posterior_measure(
            data, solver
        ).expectation
        operator_mean_zero = post_op_zero(data)

        assert np.allclose(operator_mean_zero, expected_mean_zero)

        # --- Case 2: Non-Zero Mean (Should return an AffineOperator) ---
        prior_mean = forward_problem.model_space.random()
        error_mean = forward_problem.data_space.random()

        prior_nonzero = GaussianMeasure(
            covariance=model_prior_measure.covariance, expectation=prior_mean
        )
        fp_nonzero = LinearForwardProblem(
            forward_problem.forward_operator,
            data_error_measure=GaussianMeasure(
                covariance=forward_problem.data_error_measure.covariance,
                expectation=error_mean,
            ),
        )

        inversion_affine = LinearBayesianInversion(fp_nonzero, prior_nonzero)
        post_op_affine = inversion_affine.posterior_expectation_operator(solver)

        assert isinstance(post_op_affine, AffineOperator)

        expected_mean_affine = inversion_affine.model_posterior_measure(
            data, solver
        ).expectation
        operator_mean_affine = post_op_affine(data)

        assert np.allclose(operator_mean_affine, expected_mean_affine)


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

    def test_formalism_equivalence(
        self,
        forward_problem: LinearForwardProblem,
        model_prior_measure: GaussianMeasure,
        data: np.ndarray,
    ):
        """
        Tests that both the 'data_space' and 'model_space' formalisms produce
        the exact same posterior expectation and covariance.
        """
        solver = CholeskySolver(galerkin=True)

        # 1. Solve using data space formulation
        inv_data = LinearBayesianInversion(
            forward_problem, model_prior_measure, formalism="data_space"
        )
        post_data = inv_data.model_posterior_measure(data, solver)

        # 2. Solve using model space formulation
        inv_model = LinearBayesianInversion(
            forward_problem, model_prior_measure, formalism="model_space"
        )
        post_model = inv_model.model_posterior_measure(data, solver)

        # 3. Verify Exact Equivalence
        assert np.allclose(
            forward_problem.model_space.to_components(post_data.expectation),
            forward_problem.model_space.to_components(post_model.expectation),
            atol=1e-6,
            rtol=1e-6,
        )

        assert np.allclose(
            post_data.covariance.matrix(dense=True),
            post_model.covariance.matrix(dense=True),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_invalid_formalism_initialization(
        self,
        forward_problem: LinearForwardProblem,
        model_prior_measure: GaussianMeasure,
    ):
        """
        Tests that invalid formalisms and missing inverse covariances are correctly caught.
        """
        with pytest.raises(ValueError, match="formalism must be either"):
            LinearBayesianInversion(
                forward_problem, model_prior_measure, formalism="spectral_space"
            )

        cov_only_prior = GaussianMeasure(covariance=model_prior_measure.covariance)
        with pytest.raises(ValueError, match="Prior inverse covariance must be set"):
            LinearBayesianInversion(
                forward_problem, cov_only_prior, formalism="model_space"
            )

        cov_only_error = GaussianMeasure(
            covariance=forward_problem.data_error_measure.covariance
        )
        bad_fp = LinearForwardProblem(
            forward_problem.forward_operator, data_error_measure=cov_only_error
        )
        with pytest.raises(
            ValueError, match="Data error inverse covariance must be set"
        ):
            LinearBayesianInversion(
                bad_fp, model_prior_measure, formalism="model_space"
            )

    def test_preconditioner_guards_in_model_space(
        self,
        forward_problem: LinearForwardProblem,
        model_prior_measure: GaussianMeasure,
    ):
        """
        Tests that calling data-space specific custom preconditioners raises an error
        when the inversion is configured for the model-space formalism.
        """
        inv_model = LinearBayesianInversion(
            forward_problem, model_prior_measure, formalism="model_space"
        )

        with pytest.raises(
            ValueError, match="mathematically derived for the data-space"
        ):
            inv_model.diagonal_normal_preconditioner()

        with pytest.raises(
            ValueError, match="mathematically derived for the data-space"
        ):
            inv_model.sparse_localized_preconditioner(interacting_blocks=[[0, 1]])

    def test_prior_properties(
        self,
        forward_problem: LinearForwardProblem,
        model_prior_measure: GaussianMeasure,
    ):
        """Tests that the data_prior_measure and joint_prior_measure properties are wired correctly."""
        inversion = LinearBayesianInversion(forward_problem, model_prior_measure)

        data_prior = inversion.data_prior_measure
        assert isinstance(data_prior, GaussianMeasure)
        assert data_prior.domain == forward_problem.data_space

        joint_prior = inversion.joint_prior_measure
        assert isinstance(joint_prior, GaussianMeasure)
        expected_joint_dim = (
            forward_problem.model_space.dim + forward_problem.data_space.dim
        )
        assert joint_prior.domain.dim == expected_joint_dim


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

        preconditioner = inversion.diagonal_normal_preconditioner()
        precon_diagonal = preconditioner.extract_diagonal()
        approx_normal_diagonal = 1.0 / precon_diagonal

        A = forward_problem.forward_operator.matrix(dense=True)
        Q = model_prior_measure.covariance.matrix(dense=True)
        R = forward_problem.data_error_measure.covariance.matrix(dense=True)

        exact_normal_matrix = A @ Q @ A.T + R
        exact_diagonal = np.diag(exact_normal_matrix)

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

        blocks = [[0, 1], [2], list(range(3, data_dim))]
        preconditioner = inversion.diagonal_normal_preconditioner(blocks=blocks)

        # Extract the inverted diagonal from the preconditioner and manually
        # invert it back to the original normal diagonal for verification
        precon_diagonal = preconditioner.extract_diagonal()
        approx_normal_diagonal = 1.0 / precon_diagonal

        model_space = forward_problem.model_space
        data_space = forward_problem.data_space
        A_adj = forward_problem.forward_operator.adjoint
        Q_op = model_prior_measure.covariance

        v01 = data_space.from_components(np.array([0.5, 0.5] + [0.0] * (data_dim - 2)))
        f_v01 = A_adj(v01)
        aqa_01 = model_space.inner_product(f_v01, Q_op(f_v01))

        expected_0 = aqa_01 + 1.0
        expected_1 = aqa_01 + 1.0

        assert np.isclose(approx_normal_diagonal[0], expected_0)
        assert np.isclose(approx_normal_diagonal[1], expected_1)

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

        missing_blocks = [[0, 1]] + [list(range(3, data_dim))]
        with pytest.raises(ValueError, match="must exactly partition"):
            inversion.diagonal_normal_preconditioner(blocks=missing_blocks)

        duplicate_blocks = [[0, 1], [1, 2]] + [list(range(3, data_dim))]
        with pytest.raises(ValueError, match="must exactly partition"):
            inversion.diagonal_normal_preconditioner(blocks=duplicate_blocks)

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

        data_dim = forward_problem.data_space.dim
        blocks = [list(range(data_dim))]
        preconditioner = inversion.sparse_localized_preconditioner(
            blocks, rank=data_dim
        )

        A = forward_problem.forward_operator.matrix(dense=True)
        Q = model_prior_measure.covariance.matrix(dense=True)
        R = forward_problem.data_error_measure.covariance.matrix(dense=True)

        exact_normal_matrix = A @ Q @ A.T + R
        exact_inverse_matrix = np.linalg.inv(exact_normal_matrix)

        approx_inverse_matrix = preconditioner.matrix(dense=True)
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
        data_dim = forward_problem.data_space.dim

        blocks = [list(range(0, 15)), list(range(10, 25)), list(range(20, data_dim))]
        preconditioner = inversion.sparse_localized_preconditioner(blocks, rank=10)
        result = preconditioner(data)

        assert forward_problem.data_space.is_element(result)


# =============================================================================
# New Tests for Woodbury Surrogate Preconditioner
# =============================================================================


class TestWoodburyPreconditioner:
    """Tests for the Woodbury matrix identity preconditioner."""

    def test_woodbury_exact_equivalence(
        self,
        forward_problem: LinearForwardProblem,
        model_prior_measure: GaussianMeasure,
    ):
        """
        Verifies that the Woodbury preconditioner exactly matches the inverse
        of the dense data-space normal operator.
        """
        inversion = LinearBayesianInversion(forward_problem, model_prior_measure)
        solver = LUSolver(galerkin=False)

        woodbury_precon = inversion.woodbury_data_preconditioner(solver)

        A = forward_problem.forward_operator.matrix(dense=True)
        Q = model_prior_measure.covariance.matrix(dense=True)
        R = forward_problem.data_error_measure.covariance.matrix(dense=True)

        exact_normal_matrix = A @ Q @ A.T + R
        exact_inverse_matrix = np.linalg.inv(exact_normal_matrix)
        woodbury_matrix = woodbury_precon.matrix(dense=True)

        assert np.allclose(woodbury_matrix, exact_inverse_matrix, atol=1e-8)

    def test_surrogate_woodbury_chaining(
        self,
        forward_problem: LinearForwardProblem,
        model_prior_measure: GaussianMeasure,
    ):
        """
        Verifies that the surrogate wrapper correctly chains the surrogate
        inversion and the Woodbury preconditioner extraction.
        """
        inversion = LinearBayesianInversion(forward_problem, model_prior_measure)
        alt_A = 0.5 * forward_problem.forward_operator
        solver = LUSolver(galerkin=False)

        chained_precon = inversion.surrogate_woodbury_data_preconditioner(
            solver, alternate_forward_operator=alt_A
        )

        manual_surrogate = inversion.surrogate_inversion(
            alternate_forward_operator=alt_A
        )
        manual_precon = manual_surrogate.woodbury_data_preconditioner(solver)

        assert np.allclose(
            chained_precon.matrix(dense=True), manual_precon.matrix(dense=True)
        )

    def test_woodbury_requires_data_error(self, forward_problem: LinearForwardProblem):
        """
        Verifies that the Woodbury identity fails safely if no data error measure is set.
        """
        fp_no_noise = LinearForwardProblem(forward_problem.forward_operator)
        prior = GaussianMeasure.from_standard_deviation(fp_no_noise.model_space, 1.0)
        inversion = LinearBayesianInversion(fp_no_noise, prior)
        solver = LUSolver(galerkin=False)

        with pytest.raises(ValueError, match="Data error measure must be set"):
            inversion.woodbury_data_preconditioner(solver)


# =============================================================================
# New Tests for Normal Residual Callbacks
# =============================================================================


class TestNormalResidualCallback:
    """Tests for the normal residual callback generator."""

    def test_rhs_shifting_logic(
        self, forward_problem: LinearForwardProblem, data: np.ndarray
    ):
        """
        Verifies that the right-hand side of the normal equations correctly
        shifts the observed data by the prior and error expectations.
        """
        data_space = forward_problem.data_space
        model_space = forward_problem.model_space
        A = forward_problem.forward_operator

        mu_u = model_space.random()
        mu_e = data_space.random()

        prior = GaussianMeasure.from_standard_deviation(
            model_space, 1.0, expectation=mu_u
        )
        noisy_fp = LinearForwardProblem(
            A,
            data_error_measure=GaussianMeasure.from_standard_deviation(
                data_space, 1.0, expectation=mu_e
            ),
        )
        inversion = LinearBayesianInversion(noisy_fp, prior, formalism="data_space")
        rhs = inversion.get_normal_equations_rhs(data)

        manual_rhs = data_space.subtract(data, A(mu_u))
        manual_rhs = data_space.subtract(manual_rhs, mu_e)

        assert np.allclose(
            data_space.to_components(rhs), data_space.to_components(manual_rhs)
        )

    def test_callback_generation(
        self,
        forward_problem: LinearForwardProblem,
        model_prior_measure: GaussianMeasure,
        data: np.ndarray,
    ):
        """
        Verifies that the callback generator wires the correct operator and RHS
        into the ResidualTrackingCallback.
        """
        inversion = LinearBayesianInversion(forward_problem, model_prior_measure)
        callback = inversion.normal_residual_callback(data, print_progress=False)

        assert isinstance(callback, ResidualTrackingCallback)

        expected_normal_matrix = inversion.normal_operator.matrix(dense=True)
        callback_normal_matrix = callback.operator.matrix(dense=True)
        assert np.allclose(callback_normal_matrix, expected_normal_matrix)

        assert np.allclose(
            forward_problem.data_space.to_components(callback.y),
            forward_problem.data_space.to_components(data),
        )


class TestBayesianParameterizedInversion:
    """
    Tests the parameterized surrogate generation for the Bayesian class,
    focusing on prior push-forward and dense matrix freezing.
    """

    def test_parameterized_auto_prior_push_forward(
        self, forward_problem, model_prior_measure
    ):
        """
        Verifies that the original prior is correctly projected onto the
        parameter space using the adjoint of the parameterization mapping.
        """
        inv = LinearBayesianInversion(forward_problem, model_prior_measure)
        param_space = EuclideanSpace(5)
        M_mat = np.random.randn(forward_problem.model_space.dim, param_space.dim)
        M = LinearOperator.from_matrix(param_space, forward_problem.model_space, M_mat)

        surrogate = inv.parameterized_inversion(M)

        assert isinstance(surrogate, LinearBayesianInversion)
        assert surrogate.model_space == param_space
        assert surrogate.model_prior_measure.domain == param_space

        Q_mat = model_prior_measure.covariance.matrix(dense=True)
        expected_pushed_cov = M_mat.T @ Q_mat @ M_mat
        actual_pushed_cov = surrogate.model_prior_measure.covariance.matrix(dense=True)

        assert np.allclose(actual_pushed_cov, expected_pushed_cov)

    def test_parameterized_dense_freeze(self, forward_problem, model_prior_measure):
        """
        Verifies that the dense=True flag correctly squashes the operator,
        prior, and noise measure into dense matrix objects.
        """
        inv = LinearBayesianInversion(forward_problem, model_prior_measure)

        param_space = EuclideanSpace(2)
        M_mat = np.random.randn(forward_problem.model_space.dim, param_space.dim)
        M = LinearOperator.from_matrix(param_space, forward_problem.model_space, M_mat)
        surrogate = inv.parameterized_inversion(M, dense=True)

        assert isinstance(
            surrogate.forward_problem.forward_operator, DenseMatrixLinearOperator
        )
        assert isinstance(
            surrogate.forward_problem.data_error_measure.covariance,
            DenseMatrixLinearOperator,
        )
        assert isinstance(
            surrogate.model_prior_measure.covariance, DenseMatrixLinearOperator
        )

    def test_parameterized_formalism_preservation(
        self, forward_problem, model_prior_measure
    ):
        """Ensures surrogate preserves the parent inversion's formalism."""
        inv = LinearBayesianInversion(
            forward_problem, model_prior_measure, formalism="model_space"
        )
        M = inv.model_space.identity_operator()
        surrogate = inv.parameterized_inversion(M)
        assert surrogate.formalism == "model_space"

    def test_with_formalism(self, forward_problem, model_prior_measure):
        """Tests that with_formalism returns a valid new instance."""
        inv_data = LinearBayesianInversion(
            forward_problem, model_prior_measure, formalism="data_space"
        )
        inv_model = inv_data.with_formalism("model_space")

        assert isinstance(inv_model, LinearBayesianInversion)
        assert inv_model.formalism == "model_space"
        assert inv_model is not inv_data
        assert inv_data.formalism == "data_space"

    def test_parameterized_formalism_override_auto_densifies(
        self, forward_problem, model_prior_measure
    ):
        """
        Tests that overriding the formalism to 'model_space' during parameterization
        automatically triggers the densification of the prior measure.
        """
        inv = LinearBayesianInversion(
            forward_problem, model_prior_measure, formalism="data_space"
        )
        param_space = EuclideanSpace(2)
        M = LinearOperator.from_matrix(
            param_space, forward_problem.model_space, np.random.randn(50, 2)
        )
        surrogate = inv.parameterized_inversion(M, formalism="model_space")

        assert surrogate.formalism == "model_space"
        assert isinstance(
            surrogate.model_prior_measure.covariance, DenseMatrixLinearOperator
        )


class TestWoodburyModelPreconditioner:
    """Tests for the model-space Woodbury matrix identity preconditioner."""

    def test_woodbury_model_exact_equivalence(
        self,
        forward_problem: LinearForwardProblem,
        model_prior_measure: GaussianMeasure,
    ):
        """
        Verifies that the model-space Woodbury preconditioner exactly matches
        the inverse of the dense model-space normal operator.
        """
        inversion = LinearBayesianInversion(
            forward_problem, model_prior_measure, formalism="model_space"
        )
        solver = LUSolver(galerkin=False)
        woodbury_precon = inversion.woodbury_model_preconditioner(solver)

        A = forward_problem.forward_operator.matrix(dense=True)
        Q = model_prior_measure.covariance.matrix(dense=True)
        R = forward_problem.data_error_measure.covariance.matrix(dense=True)
        Q_inv = np.linalg.inv(Q)
        R_inv = np.linalg.inv(R)

        exact_normal_matrix = Q_inv + A.T @ R_inv @ A
        exact_inverse_matrix = np.linalg.inv(exact_normal_matrix)
        woodbury_matrix = woodbury_precon.matrix(dense=True)

        assert np.allclose(woodbury_matrix, exact_inverse_matrix, atol=1e-8)

    def test_surrogate_woodbury_model_chaining(
        self,
        forward_problem: LinearForwardProblem,
        model_prior_measure: GaussianMeasure,
    ):
        """
        Verifies that the surrogate wrapper correctly chains the surrogate
        inversion and the model-space Woodbury preconditioner extraction.
        """
        inversion = LinearBayesianInversion(forward_problem, model_prior_measure)
        alt_A = 0.5 * forward_problem.forward_operator
        solver = LUSolver(galerkin=False)

        chained_precon = inversion.surrogate_woodbury_model_preconditioner(
            solver, alternate_forward_operator=alt_A
        )
        manual_surrogate = inversion.surrogate_inversion(
            alternate_forward_operator=alt_A
        )
        manual_precon = manual_surrogate.woodbury_model_preconditioner(solver)

        assert np.allclose(
            chained_precon.matrix(dense=True), manual_precon.matrix(dense=True)
        )

    def test_woodbury_model_requires_data_error(
        self, forward_problem: LinearForwardProblem
    ):
        """
        Verifies that the model-space Woodbury identity fails safely if no data error measure is set.
        """
        fp_no_noise = LinearForwardProblem(forward_problem.forward_operator)
        prior = GaussianMeasure.from_standard_deviation(fp_no_noise.model_space, 1.0)
        inversion = LinearBayesianInversion(fp_no_noise, prior)
        solver = LUSolver(galerkin=False)

        with pytest.raises(ValueError, match="Data error measure must be set"):
            inversion.woodbury_model_preconditioner(solver)


# =============================================================================
# New Tests for Evidence & Diagnostics
# =============================================================================


class TestMahalanobisEvidenceTerm:
    """Tests for the data-dependent Mahalanobis term of the Bayesian Evidence."""

    def test_mahalanobis_exact_dense(
        self,
        forward_problem: LinearForwardProblem,
        model_prior_measure: GaussianMeasure,
        data: np.ndarray,
    ):
        """
        Verifies that the matrix-free Mahalanobis term exactly matches
        the dense analytical calculation for a zero-mean problem.
        """
        solver = CholeskySolver(galerkin=True)
        inversion = LinearBayesianInversion(
            forward_problem, model_prior_measure, formalism="data_space"
        )
        actual_mahalanobis = inversion.mahalanobis_evidence_term(data, solver)

        A = forward_problem.forward_operator.matrix(dense=True)
        Q = model_prior_measure.covariance.matrix(dense=True)
        R = forward_problem.data_error_measure.covariance.matrix(dense=True)

        exact_normal_matrix = A @ Q @ A.T + R
        exact_inverse_matrix = np.linalg.inv(exact_normal_matrix)
        expected_mahalanobis = data.T @ exact_inverse_matrix @ data

        assert np.isclose(actual_mahalanobis, expected_mahalanobis)

    def test_mahalanobis_non_zero_mean(
        self,
        forward_problem: LinearForwardProblem,
        model_prior_measure: GaussianMeasure,
        data: np.ndarray,
    ):
        """
        Verifies that shifting by the prior and noise expectations is handled correctly.
        """
        solver = CholeskySolver(galerkin=True)

        mu_u = forward_problem.model_space.random()
        mu_e = forward_problem.data_space.random()

        prior_nonzero = GaussianMeasure(
            covariance=model_prior_measure.covariance, expectation=mu_u
        )
        fp_nonzero = LinearForwardProblem(
            forward_problem.forward_operator,
            data_error_measure=GaussianMeasure(
                covariance=forward_problem.data_error_measure.covariance,
                expectation=mu_e,
            ),
        )

        inversion = LinearBayesianInversion(
            fp_nonzero, prior_nonzero, formalism="data_space"
        )
        actual_mahalanobis = inversion.mahalanobis_evidence_term(data, solver)

        A = forward_problem.forward_operator.matrix(dense=True)
        Q = model_prior_measure.covariance.matrix(dense=True)
        R = forward_problem.data_error_measure.covariance.matrix(dense=True)

        exact_normal_matrix = A @ Q @ A.T + R
        exact_inverse_matrix = np.linalg.inv(exact_normal_matrix)

        v_data = data - (A @ mu_u) - mu_e
        expected_mahalanobis = v_data.T @ exact_inverse_matrix @ v_data

        assert np.isclose(actual_mahalanobis, expected_mahalanobis)

    def test_formalism_equivalence(
        self,
        forward_problem: LinearForwardProblem,
        model_prior_measure: GaussianMeasure,
        data: np.ndarray,
    ):
        """
        Tests that both the 'data_space' and Woodbury 'model_space' formalisms
        produce the exact same Mahalanobis scalar.
        """
        solver = CholeskySolver(galerkin=True)

        inv_data = LinearBayesianInversion(
            forward_problem, model_prior_measure, formalism="data_space"
        )
        mahalanobis_data = inv_data.mahalanobis_evidence_term(data, solver)

        inv_model = LinearBayesianInversion(
            forward_problem, model_prior_measure, formalism="model_space"
        )
        mahalanobis_model = inv_model.mahalanobis_evidence_term(data, solver)

        assert np.isclose(mahalanobis_data, mahalanobis_model)

    def test_mahalanobis_requires_data_error(
        self, forward_problem: LinearForwardProblem
    ):
        """
        Verifies that the method fails safely if no data error measure is set.
        """
        fp_no_noise = LinearForwardProblem(forward_problem.forward_operator)
        prior = GaussianMeasure.from_standard_deviation(fp_no_noise.model_space, 1.0)
        inversion = LinearBayesianInversion(fp_no_noise, prior)
        solver = LUSolver(galerkin=False)
        dummy_data = fp_no_noise.data_space.zero

        with pytest.raises(ValueError, match="Data error measure must be set"):
            inversion.mahalanobis_evidence_term(dummy_data, solver)


class TestBayesianEvidence:
    """Tests for the Log-Determinant SLQ estimator and full Log-Evidence calculations."""

    def test_estimate_log_determinant_data_space(
        self,
        forward_problem: LinearForwardProblem,
        model_prior_measure: GaussianMeasure,
    ):
        """
        Verifies that the SLQ estimator converges to the neighborhood of the exact
        dense log-determinant of the data-space normal operator.
        """
        inversion = LinearBayesianInversion(
            forward_problem, model_prior_measure, formalism="data_space"
        )

        # 1. Exact Dense Calculation
        A = forward_problem.forward_operator.matrix(dense=True)
        Q = model_prior_measure.covariance.matrix(dense=True)
        R = forward_problem.data_error_measure.covariance.matrix(dense=True)

        exact_normal_matrix = A @ Q @ A.T + R
        sign, exact_log_det = np.linalg.slogdet(exact_normal_matrix)
        assert sign > 0

        # 2. SLQ Matrix-Free Estimation
        # Using lanczos_degree = dim ensures the inner Krylov subspace is exact.
        # The Hutchinson outer-loop still has variance, so we use a statistical tolerance.
        dim = forward_problem.data_space.dim
        np.random.seed(42)  # Fix seed for stable test variance
        approx_log_det = inversion.estimate_log_determinant(
            operator_type="data_space",
            size_estimate=dim,
            method="fixed",
            lanczos_degree=dim,
            lanczos_rtol=None,  # Force exact Lanczos
        )

        assert np.isclose(approx_log_det, exact_log_det, rtol=0.05)

    def test_estimate_log_determinant_model_space(
        self,
        forward_problem: LinearForwardProblem,
        model_prior_measure: GaussianMeasure,
    ):
        """
        Verifies that the SLQ estimator converges to the neighborhood of the exact
        dense log-determinant of the model-space normal operator.
        """
        # Force model_space formalism to trigger prior inverse requirements
        solver = CholeskySolver(galerkin=True)
        prior = model_prior_measure.with_regularized_inverse(solver, damping=0.1)
        error = forward_problem.data_error_measure.with_regularized_inverse(
            solver, damping=0.1
        )

        fp_invertible = LinearForwardProblem(
            forward_problem.forward_operator, data_error_measure=error
        )

        inversion = LinearBayesianInversion(
            fp_invertible, prior, formalism="model_space"
        )

        # 1. Exact Dense Calculation
        A = fp_invertible.forward_operator.matrix(dense=True)
        Q_inv = prior.inverse_covariance.matrix(dense=True)
        R_inv = error.inverse_covariance.matrix(dense=True)

        exact_normal_matrix = Q_inv + A.T @ R_inv @ A
        sign, exact_log_det = np.linalg.slogdet(exact_normal_matrix)
        assert sign > 0

        # 2. SLQ Matrix-Free Estimation
        dim = fp_invertible.model_space.dim
        np.random.seed(42)
        approx_log_det = inversion.estimate_log_determinant(
            operator_type="model_space",
            size_estimate=dim,
            method="fixed",
            lanczos_degree=dim,
            lanczos_rtol=None,
        )

        assert np.isclose(approx_log_det, exact_log_det, rtol=0.05)

    def test_log_evidence_exact_dense(
        self,
        forward_problem: LinearForwardProblem,
        model_prior_measure: GaussianMeasure,
        data: np.ndarray,
    ):
        """
        Verifies that the full log-evidence computation closely matches the
        analytical likelihood of the data under the marginal distribution.
        """
        solver = CholeskySolver(galerkin=True)
        inversion = LinearBayesianInversion(
            forward_problem, model_prior_measure, formalism="data_space"
        )

        # 1. Exact Dense Calculation
        A = forward_problem.forward_operator.matrix(dense=True)
        Q = model_prior_measure.covariance.matrix(dense=True)
        R = forward_problem.data_error_measure.covariance.matrix(dense=True)

        exact_normal_matrix = A @ Q @ A.T + R

        # Using Scipy's multivariate normal to compute exact log p(d)
        from scipy.stats import multivariate_normal

        exact_marginal = multivariate_normal(
            mean=np.zeros(forward_problem.data_space.dim), cov=exact_normal_matrix
        )
        expected_evidence = exact_marginal.logpdf(data)

        # 2. Matrix-Free Evidence Calculation
        dim = forward_problem.data_space.dim
        np.random.seed(42)
        actual_evidence = inversion.log_evidence(
            data,
            solver,
            size_estimate=dim,
            method="fixed",
            lanczos_degree=dim,
            lanczos_rtol=None,
        )

        assert np.isclose(actual_evidence, expected_evidence, rtol=0.05)

    def test_log_evidence_dynamic_convergence(
        self,
        forward_problem: LinearForwardProblem,
        model_prior_measure: GaussianMeasure,
        data: np.ndarray,
    ):
        """
        Verifies that the 'variable' method successfully evaluates without errors
        and produces a statistically sound estimate of the log-evidence.
        """
        solver = CholeskySolver(galerkin=True)
        inversion = LinearBayesianInversion(
            forward_problem, model_prior_measure, formalism="data_space"
        )

        dim = forward_problem.data_space.dim
        np.random.seed(42)

        # Running the dynamic SLQ logic
        actual_evidence = inversion.log_evidence(
            data,
            solver,
            size_estimate=10,
            method="variable",
            max_samples=dim * 2,  # Let Hutchinson run a bit to ensure stability
            rtol=1e-2,
            lanczos_degree=dim,
            lanczos_rtol=1e-3,
        )

        # Because it is stochastic, we just want to ensure it completes and returns a float
        assert isinstance(actual_evidence, float)

    def test_log_evidence_model_space_sylvester_identity(
        self,
        forward_problem: LinearForwardProblem,
        model_prior_measure: GaussianMeasure,
        data: np.ndarray,
    ):
        """
        Verifies that the log-evidence computed in the 'model_space' formalism
        (which relies on Sylvester's determinant identity: ln|Nm| + ln|Q| + ln|R|)
        matches the exact dense analytical likelihood.
        """
        solver = CholeskySolver(galerkin=True)

        # Force prior and error to have strict inverses for model_space formulation
        prior = model_prior_measure.with_regularized_inverse(solver, damping=0.1)
        error = forward_problem.data_error_measure.with_regularized_inverse(
            solver, damping=0.1
        )
        fp_invertible = LinearForwardProblem(
            forward_problem.forward_operator, data_error_measure=error
        )

        inversion = LinearBayesianInversion(
            fp_invertible, prior, formalism="model_space"
        )

        # 1. Exact Dense Calculation (Data Space Marginal)
        A = fp_invertible.forward_operator.matrix(dense=True)
        Q = prior.covariance.matrix(dense=True)
        R = error.covariance.matrix(dense=True)

        exact_normal_matrix = A @ Q @ A.T + R

        from scipy.stats import multivariate_normal

        exact_marginal = multivariate_normal(
            mean=np.zeros(forward_problem.data_space.dim), cov=exact_normal_matrix
        )
        expected_evidence = exact_marginal.logpdf(data)

        # 2. Matrix-Free Evidence Calculation using Sylvester's Identity
        # We use a high sample count to tightly bound the stochastic error
        dim_data = forward_problem.data_space.dim
        dim_model = forward_problem.model_space.dim
        max_dim = max(dim_data, dim_model)

        np.random.seed(42)
        actual_evidence = inversion.log_evidence(
            data,
            solver,
            size_estimate=max_dim,
            method="fixed",
            lanczos_degree=max_dim,
            lanczos_rtol=None,
        )

        # Check equivalence
        assert np.isclose(actual_evidence, expected_evidence, rtol=0.05)

    def test_estimate_log_determinant_cross_formalism(
        self,
        forward_problem: LinearForwardProblem,
        model_prior_measure: GaussianMeasure,
    ):
        """
        Verifies the bugfix where calling estimate_log_determinant for 'data_space'
        on a 'model_space' inversion correctly instantiates the internal surrogate
        and avoids dimensionality mismatch crashes.
        """
        solver = CholeskySolver(galerkin=True)
        prior = model_prior_measure.with_regularized_inverse(solver, damping=0.1)
        error = forward_problem.data_error_measure.with_regularized_inverse(
            solver, damping=0.1
        )
        fp_invertible = LinearForwardProblem(
            forward_problem.forward_operator, data_error_measure=error
        )

        # Initialize in model_space
        inversion = LinearBayesianInversion(
            fp_invertible, prior, formalism="model_space"
        )

        # Request data_space determinant explicitly
        dim = fp_invertible.data_space.dim
        np.random.seed(42)
        approx_log_det = inversion.estimate_log_determinant(
            operator_type="data_space",
            size_estimate=dim,
            method="fixed",
            lanczos_degree=dim,
            lanczos_rtol=None,
        )

        # Verify against exact dense data-space determinant
        A = fp_invertible.forward_operator.matrix(dense=True)
        Q = prior.covariance.matrix(dense=True)
        R = error.covariance.matrix(dense=True)
        exact_normal_matrix = A @ Q @ A.T + R
        sign, exact_log_det = np.linalg.slogdet(exact_normal_matrix)

        assert np.isclose(approx_log_det, exact_log_det, rtol=0.05)
