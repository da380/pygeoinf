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
from pygeoinf.subspaces import AffineSubspace, LinearSubspace
from pygeoinf.linear_optimisation import (
    LinearLeastSquaresInversion,
    LinearMinimumNormInversion,
    ConstrainedLinearLeastSquaresInversion,
    ConstrainedLinearMinimumNormInversion,
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
# Tests for ConstrainedLinearLeastSquaresInversion
# =============================================================================


class TestConstrainedLinearLeastSquaresInversion:
    """Tests for the subspace-constrained solver."""

    def test_constrained_solver_explicit_geometry(self):
        """
        Test using a manually constructed geometric constraint.
        Problem: d = u (Identity), subject to u lying on plane z=1.
        Data: (10, 10, 10).
        Expected: (10, 10, 1).
        """
        r3 = EuclideanSpace(dim=3)

        # 1. Forward Problem: Identity
        I = r3.identity_operator()
        fp = LinearForwardProblem(I)

        # 2. Constraint: Affine plane z=1
        # Tangent space is XY plane (spanned by e1, e2)
        e1 = r3.basis_vector(0)
        e2 = r3.basis_vector(1)

        # Use the new API: LinearSubspace.from_basis
        tangent_space = LinearSubspace.from_basis(r3, [e1, e2])

        # Translation to z=1
        z_offset = r3.from_components(np.array([0.0, 0.0, 1.0]))

        # Construct affine subspace manually
        constraint = AffineSubspace(tangent_space.projector, translation=z_offset)

        # 3. Solve
        solver = ConstrainedLinearLeastSquaresInversion(fp, constraint)

        # Very small damping to mimic pure projection
        ls_op = solver.least_squares_operator(1e-8, CholeskySolver(galerkin=True))

        data = r3.from_components(np.array([10.0, 10.0, 10.0]))
        u_sol = ls_op(data)

        # 4. Verify
        expected = r3.from_components(np.array([10.0, 10.0, 1.0]))
        assert np.allclose(u_sol, expected, atol=1e-5)
        assert constraint.is_element(u_sol)

    def test_constrained_solver_implicit_equation(self):
        """
        Test using a constraint defined by a linear equation B(u) = w.
        Problem: Minimize ||u - data||^2 subject to mean(u) = 2.
        """
        # Setup: 5D space
        domain = EuclideanSpace(5)

        # Forward problem: Identity (denoising)
        fp = LinearForwardProblem(domain.identity_operator())

        # Constraint Operator B: u -> mean(u) (maps R5 -> R1)
        # B = [0.2, 0.2, 0.2, 0.2, 0.2]
        codomain = EuclideanSpace(1)
        ones = np.ones((1, 5)) / 5.0
        B = LinearOperator.from_matrix(domain, codomain, ones)

        # Target value w = 2.0
        w = codomain.from_components(np.array([2.0]))

        # Create Subspace implicitly
        constraint = AffineSubspace.from_linear_equation(B, w)

        # Solve
        solver = ConstrainedLinearLeastSquaresInversion(fp, constraint)
        ls_op = solver.least_squares_operator(1e-8, CholeskySolver(galerkin=True))

        # Data: Random vector
        data_vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # Mean is 3.0
        data = domain.from_components(data_vec)

        u_sol = ls_op(data)
        u_vec = domain.to_components(u_sol)

        # Verification 1: Constraint Satisfaction
        # The mean of the solution should be exactly 2.0
        assert np.isclose(np.mean(u_vec), 2.0)

        # Verification 2: Geometry
        # The solution should be the projection of 'data' onto the plane sum(u)=10.
        # This corresponds to subtracting the difference in means.
        # Diff = 3.0 - 2.0 = 1.0.  u_expected = data - 1.0
        expected_vec = data_vec - 1.0
        assert np.allclose(u_vec, expected_vec, atol=1e-5)


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


# =============================================================================
# Tests for ConstrainedLinearMinimumNormInversion
# =============================================================================


class TestConstrainedLinearMinimumNormInversion:
    """Tests for the constrained discrepancy principle solver."""

    def test_constrained_min_norm_simple(self):
        """
        Test problem:
        Data matches the constrained minimum norm solution to verify the
        zero-perturbation behavior.

        Constraint: Mean(u) = 10.
        u_base (solution closest to origin) is [10, 10, 10, 10, 10].

        If we set Data = [10, 10, ...], then u_base fits the data perfectly.
        The reduced problem sees data_tilde = d - A(u_base) = 0.
        It should return w=0, so u = u_base.
        """
        domain = EuclideanSpace(5)
        # Forward operator is Identity
        fp = LinearForwardProblem(
            domain.identity_operator(),
            data_error_measure=GaussianMeasure.from_standard_deviation(domain, 1.0),
        )

        # Constraint: Mean(u) = 10
        codomain = EuclideanSpace(1)
        ones = np.ones((1, 5)) / 5.0
        B = LinearOperator.from_matrix(domain, codomain, ones)
        w = codomain.from_components(np.array([10.0]))
        constraint = AffineSubspace.from_linear_equation(B, w)

        solver = ConstrainedLinearMinimumNormInversion(fp, constraint)
        op = solver.minimum_norm_operator(
            CholeskySolver(galerkin=True), significance_level=0.5
        )

        # Set data equal to the expected base solution to ensure fit is possible
        data = domain.from_components(np.full(5, 10.0))
        u_sol = op(data)

        # The solution should be the uniform vector [10, ...]
        expected = np.full(5, 10.0)
        u_vec = domain.to_components(u_sol)

        assert np.allclose(u_vec, expected)
        assert constraint.is_element(u_sol)
