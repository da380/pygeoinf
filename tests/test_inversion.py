"""
Tests for the Inversion base class.
"""

import pytest
import numpy as np
from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.linear_operators import LinearOperator
from pygeoinf.gaussian_measure import GaussianMeasure
from pygeoinf.forward_problem import LinearForwardProblem
from pygeoinf.inversion import Inversion, LinearInversion
from pygeoinf.linear_bayesian import LinearBayesianInversion
from pygeoinf.linear_optimisation import LinearLeastSquaresInversion
from pygeoinf.linear_operators import DenseMatrixLinearOperator


# =============================================================================
# Fixtures for Different Forward Problem Setups
# =============================================================================


@pytest.fixture
def forward_problem_no_error() -> LinearForwardProblem:
    """A forward problem with no data error measure."""
    model_space = EuclideanSpace(2)
    data_space = EuclideanSpace(2)
    op = LinearOperator.from_matrix(model_space, data_space, np.eye(2))
    return LinearForwardProblem(op)


@pytest.fixture
def forward_problem_with_error_no_inv_cov() -> LinearForwardProblem:
    """A forward problem with a data error measure that lacks an inverse covariance."""
    model_space = EuclideanSpace(2)
    data_space = EuclideanSpace(2)
    op = LinearOperator.from_matrix(model_space, data_space, np.eye(2))
    # Create a measure with a covariance factor, but no inverse factor
    cov_factor = LinearOperator.from_matrix(
        EuclideanSpace(2), data_space, 0.1 * np.eye(2)
    )
    error_measure = GaussianMeasure(covariance_factor=cov_factor)
    return LinearForwardProblem(op, data_error_measure=error_measure)


@pytest.fixture
def forward_problem_with_inv_cov() -> LinearForwardProblem:
    """A forward problem with a data error measure that has an inverse covariance."""
    model_space = EuclideanSpace(2)
    data_space = EuclideanSpace(2)
    op = LinearOperator.from_matrix(model_space, data_space, np.eye(2))
    error_measure = GaussianMeasure.from_standard_deviation(data_space, 0.1)
    return LinearForwardProblem(op, data_error_measure=error_measure)


# A simple concrete subclass for testing the abstract Inversion class
class ConcreteInversion(Inversion):
    pass


# =============================================================================
# Tests for the Inversion Class
# =============================================================================


class TestInversion:
    """A suite of tests for the Inversion base class."""

    def test_initialization(self, forward_problem_no_error: LinearForwardProblem):
        """Tests that the Inversion class can be instantiated."""
        inv = ConcreteInversion(forward_problem_no_error)
        assert inv is not None
        assert inv.forward_problem == forward_problem_no_error

    def test_assert_data_error_measure_raises(
        self, forward_problem_no_error: LinearForwardProblem
    ):
        """
        Tests that assert_data_error_measure raises an error when no
        measure is set.
        """
        inv = ConcreteInversion(forward_problem_no_error)
        with pytest.raises(AttributeError):
            inv.assert_data_error_measure()

    def test_assert_data_error_measure_passes(
        self, forward_problem_with_inv_cov: LinearForwardProblem
    ):
        """
        Tests that assert_data_error_measure passes when a measure is set.
        """
        inv = ConcreteInversion(forward_problem_with_inv_cov)
        try:
            inv.assert_data_error_measure()
        except AttributeError:
            pytest.fail("assert_data_error_measure raised an error unexpectedly.")

    def test_assert_inverse_data_covariance_raises_no_measure(
        self, forward_problem_no_error: LinearForwardProblem
    ):
        """
        Tests that assert_inverse_data_covariance raises an error when
        no data error measure is set at all.
        """
        inv = ConcreteInversion(forward_problem_no_error)
        with pytest.raises(AttributeError):
            inv.assert_inverse_data_covariance()

    def test_assert_inverse_data_covariance_raises_no_inv_cov(
        self, forward_problem_with_error_no_inv_cov: LinearForwardProblem
    ):
        """
        Tests that assert_inverse_data_covariance raises an error when the
        measure exists but has no inverse covariance.
        """
        inv = ConcreteInversion(forward_problem_with_error_no_inv_cov)
        with pytest.raises(AttributeError):
            inv.assert_inverse_data_covariance()

    def test_assert_inverse_data_covariance_passes(
        self, forward_problem_with_inv_cov: LinearForwardProblem
    ):
        """
        Tests that assert_inverse_data_covariance passes when an inverse
        covariance is available.
        """
        inv = ConcreteInversion(forward_problem_with_inv_cov)
        try:
            inv.assert_inverse_data_covariance()
        except AttributeError:
            pytest.fail("assert_inverse_data_covariance raised an error unexpectedly.")


class ConcreteLinearInversion(LinearInversion):
    """A simple concrete subclass for testing LinearInversion."""

    pass


class TestLinearInversion:
    """A suite of tests for the LinearInversion base class."""

    def test_measure_pass_throughs(
        self, forward_problem_with_inv_cov: LinearForwardProblem
    ):
        """Tests that the statistical mapping methods successfully delegate to the forward problem."""
        inv = ConcreteLinearInversion(forward_problem_with_inv_cov)
        model_space = forward_problem_with_inv_cov.model_space

        model = model_space.random()
        prior = GaussianMeasure.from_standard_deviation(model_space, 1.0)

        # We rely on test_forward_problem for the math; here we just ensure the wiring works.
        dm = inv.data_measure_from_model(model)
        assert isinstance(dm, GaussianMeasure)

        dpm = inv.data_measure_from_model_measure(prior)
        assert isinstance(dpm, GaussianMeasure)

        jm = inv.joint_measure(prior)
        assert isinstance(jm, GaussianMeasure)

    def test_formalism_logic(self, forward_problem_no_error: LinearForwardProblem):
        """Tests that formalism is correctly stored and validated in the base class."""
        # Test default
        inv_default = ConcreteLinearInversion(forward_problem_no_error)
        assert inv_default.formalism == "data_space"

        # Test explicit
        inv_model = ConcreteLinearInversion(
            forward_problem_no_error, formalism="model_space"
        )
        assert inv_model.formalism == "model_space"

        # Test validation
        with pytest.raises(ValueError, match="formalism must be either"):
            ConcreteLinearInversion(forward_problem_no_error, formalism="wrong_space")

    def test_base_parameterized_inversion_type(
        self, forward_problem_no_error: LinearForwardProblem
    ):
        """
        Tests that the base parameterized_inversion method preserves
        class type and formalism.
        """
        inv = ConcreteLinearInversion(forward_problem_no_error, formalism="model_space")

        # Define a dummy parameterization (Identity)
        param_op = inv.model_space.identity_operator()

        # Create the surrogate
        surrogate = inv.parameterized_inversion(param_op)

        # Verify it is the same concrete class and has the same formalism
        assert isinstance(surrogate, ConcreteLinearInversion)
        assert surrogate.formalism == "model_space"

    def test_with_formalism_raises_not_implemented(
        self, forward_problem_no_error: LinearForwardProblem
    ):
        """Tests that the base with_formalism method enforces subclass implementation."""
        inv = ConcreteLinearInversion(forward_problem_no_error)
        with pytest.raises(NotImplementedError):
            inv.with_formalism("model_space")

    def test_parameterized_inversion_formalism_override(
        self, forward_problem_no_error: LinearForwardProblem
    ):
        """Tests that parameterized_inversion correctly accepts a formalism override."""
        inv = ConcreteLinearInversion(forward_problem_no_error, formalism="data_space")
        param_op = inv.model_space.identity_operator()

        # Override to model_space
        surrogate = inv.parameterized_inversion(param_op, formalism="model_space")

        assert surrogate.formalism == "model_space"
        # Ensure original was not mutated
        assert inv.formalism == "data_space"


class TestParameterizedInversions:
    """
    Tests the parameterized surrogate generation for concrete inversion classes.
    """

    def test_bayesian_parameterization_auto_prior(self, forward_problem_with_inv_cov):
        """Tests the automatic prior push-forward in Bayesian parameterization."""
        # 1. Setup a standard Bayesian inversion
        prior = GaussianMeasure.from_standard_deviation(
            forward_problem_with_inv_cov.model_space, 1.0
        )
        inv = LinearBayesianInversion(forward_problem_with_inv_cov, prior)

        # 2. Define a parameterization mapping R^1 -> Model Space
        param_space = EuclideanSpace(1)
        # Vector in model space that our 1D parameter scales
        template_vec = inv.model_space.random()
        param_op = LinearOperator.from_vector(inv.model_space, template_vec).adjoint
        # param_op maps Euclidean(1) -> ModelSpace

        # 3. Create surrogate without providing a prior
        surrogate = inv.parameterized_inversion(param_op)

        assert isinstance(surrogate, LinearBayesianInversion)
        assert surrogate.model_prior_measure.domain == param_space

        # Verify the pushed-forward prior variance matches <m, Q m>
        # where m is the adjoint of our parameterization (the template vector)
        expected_var = inv.model_prior_measure.directional_variance(template_vec)
        actual_var = surrogate.model_prior_measure.covariance.matrix(dense=True)[0, 0]
        assert np.isclose(actual_var, expected_var)

    def test_bayesian_parameterization_dense_freeze(self, forward_problem_with_inv_cov):
        """Tests that the dense=True flag squashes all operators and measures."""
        prior = GaussianMeasure.from_standard_deviation(
            forward_problem_with_inv_cov.model_space, 1.0
        )
        inv = LinearBayesianInversion(forward_problem_with_inv_cov, prior)

        param_op = (
            inv.model_space.identity_operator()
        )  # Mapping to itself for simplicity

        # Create a dense surrogate
        surrogate = inv.parameterized_inversion(param_op, dense=True)

        # Verify everything is a DenseMatrixLinearOperator
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

    def test_least_squares_parameterization_inheritance(self, forward_problem_no_error):
        """Tests that LinearLeastSquaresInversion correctly inherits and uses the base method."""
        inv = LinearLeastSquaresInversion(
            forward_problem_no_error, formalism="data_space"
        )

        param_op = inv.model_space.identity_operator()
        surrogate = inv.parameterized_inversion(param_op)

        assert isinstance(surrogate, LinearLeastSquaresInversion)
        assert surrogate.formalism == "data_space"
