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
from pygeoinf.affine_operators import AffineOperator
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

        # Verify that the architecture returns a valid Operator type
        from pygeoinf.linear_operators import LinearOperator

        assert isinstance(lsq_operator, (AffineOperator, LinearOperator))

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

        # Use the API: LinearSubspace.from_basis
        tangent_space = LinearSubspace.from_basis(r3, [e1, e2])

        # Translation to z=1
        z_offset = r3.from_components(np.array([0.0, 0.0, 1.0]))

        # Construct affine subspace manually
        constraint = AffineSubspace(tangent_space.projector, translation=z_offset)

        # 3. Solve
        solver = ConstrainedLinearLeastSquaresInversion(fp, constraint)

        # Very small damping to mimic pure projection
        ls_op = solver.least_squares_operator(1e-8, CholeskySolver(galerkin=True))

        # Verify the highly optimized composition resolved to an AffineOperator
        assert isinstance(ls_op, AffineOperator)

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

        # Verify operator structure
        assert isinstance(ls_op, AffineOperator)

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

    def test_minimum_norm_derivative(
        self, forward_problem: LinearForwardProblem, synthetic_data: np.ndarray
    ):
        """
        Verifies the analytical Fréchet derivative of the discrepancy search,
        including its forward action against finite differences and its adjoint.
        """
        solver = CholeskySolver(galerkin=True)

        # We tighten the bracketing tolerance here so the numerical noise of the
        # solver doesn't wash out the finite difference approximation step.
        min_norm_inversion = LinearMinimumNormInversion(forward_problem)
        min_norm_operator = min_norm_inversion.minimum_norm_operator(
            solver, significance_level=0.95, rtol=1e-10, atol=1e-12
        )

        # 1. Verify the derivative is extracted properly
        derivative_op = min_norm_operator.derivative(synthetic_data)
        assert isinstance(derivative_op, LinearOperator)

        # 2. Test the adjoint mapping of the derivative explicitly using the
        # linear operator's built-in check (verifies <Du, v> == <u, D*v>)
        derivative_op.check(n_checks=3)

        # 3. Manual finite difference check for the non-linear action
        # We test locally around synthetic_data to ensure we are in the
        # active constraint regime (damping > 0).
        data_space = forward_problem.data_space
        model_space = forward_problem.model_space

        v = data_space.random()
        h = 1e-5

        data_plus = data_space.add(synthetic_data, data_space.multiply(h, v))

        u_base = min_norm_operator(synthetic_data)
        u_plus = min_norm_operator(data_plus)

        fd_approx = model_space.multiply(1.0 / h, model_space.subtract(u_plus, u_base))

        exact_dir_deriv = derivative_op(v)

        assert np.allclose(
            model_space.to_components(fd_approx),
            model_space.to_components(exact_dir_deriv),
            rtol=1e-3,
            atol=1e-3,
        )


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

    def test_constrained_minimum_norm_derivative(self):
        """
        Verifies the Fréchet derivative of the constrained discrepancy search,
        ensuring it matches finite differences and that the resulting model
        perturbations lie strictly within the constraint's tangent space.
        """
        domain = EuclideanSpace(5)
        # Identity forward operator with standard noise
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

        # Tight tolerances to ensure stable finite difference comparison
        op = solver.minimum_norm_operator(
            CholeskySolver(galerkin=True),
            significance_level=0.95,
            rtol=1e-10,
            atol=1e-12,
        )

        # Use noisy data to ensure the discrepancy principle actively penalizes
        data_vec = np.array([15.0, 5.0, 20.0, 0.0, 10.0])
        data = domain.from_components(data_vec)

        # 1. Extract derivative
        derivative_op = op.derivative(data)
        assert isinstance(derivative_op, LinearOperator)

        # 2. Check Adjoint internally (Riesz representation)
        derivative_op.check(n_checks=3)

        # 3. Verify Tangent Space Geometry
        # The constraint is mean(u) = 10, so the tangent space requires mean(du) = 0.
        v = domain.random()
        exact_dir_deriv = derivative_op(v)
        du_vec = domain.to_components(exact_dir_deriv)
        assert np.isclose(np.mean(du_vec), 0.0, atol=1e-7)

        # 4. Manual Finite Difference Check
        h = 1e-5
        data_plus = domain.add(data, domain.multiply(h, v))

        u_base = op(data)
        u_plus = op(data_plus)

        fd_approx = domain.multiply(1.0 / h, domain.subtract(u_plus, u_base))

        assert np.allclose(
            domain.to_components(fd_approx), du_vec, rtol=1e-3, atol=1e-3
        )

    def test_constraint_value_mapping_and_derivative(self):
        """
        Verifies the non-linear mapping from constraint value 'w' to the
        constrained minimum norm solution 'u(w)'.
        """
        domain = EuclideanSpace(5)

        # FIX: We use a std dev of 2.0. If it were 1.0, the minimum possible
        # chi-squared for a mean-shift of 2.0 would be 20.0, which is higher
        # than the 95% critical value (~11.07), making the discrepancy principle
        # mathematically impossible and exploding the analytical derivative!
        fp = LinearForwardProblem(
            domain.identity_operator(),
            data_error_measure=GaussianMeasure.from_standard_deviation(domain, 2.0),
        )

        # Constraint operator B: Mean(u) = w
        codomain = EuclideanSpace(1)
        ones = np.ones((1, 5)) / 5.0
        B = LinearOperator.from_matrix(domain, codomain, ones)

        # Initial constraint value w = 10.0
        w_initial = codomain.from_components(np.array([10.0]))
        constraint = AffineSubspace.from_linear_equation(B, w_initial)

        solver = ConstrainedLinearMinimumNormInversion(fp, constraint)

        # Fixed dataset
        data_vec = np.array([15.0, 5.0, 20.0, 0.0, 10.0])
        data = domain.from_components(data_vec)

        # Get the w -> u(w) mapping. Tight tolerances for finite difference stability.
        w_mapping = solver.constraint_value_mapping(
            data,
            CholeskySolver(galerkin=True),
            significance_level=0.95,
            rtol=1e-10,
            atol=1e-12,
        )

        # 1. Verify the Forward Mapping
        w_new_val = np.array([12.0])
        w_new = codomain.from_components(w_new_val)
        u_sol = w_mapping(w_new)

        assert np.allclose(codomain.to_components(B(u_sol)), w_new_val, atol=1e-7)

        # 2. Extract the Derivative at w_new
        derivative_op = w_mapping.derivative(w_new)
        assert isinstance(derivative_op, LinearOperator)

        # 3. Verify the Adjoint
        derivative_op.check(n_checks=3)

        # 4. Manual Finite Difference Check
        dw = codomain.random()
        h = 1e-5
        w_plus = codomain.add(w_new, codomain.multiply(h, dw))

        u_plus = w_mapping(w_plus)

        fd_approx = domain.multiply(1.0 / h, domain.subtract(u_plus, u_sol))

        exact_dir_deriv = derivative_op(dw)

        assert np.allclose(
            domain.to_components(fd_approx),
            domain.to_components(exact_dir_deriv),
            rtol=1e-3,
            atol=1e-3,
        )
