"""
Tests for the linear_optimisation module.
"""

import pytest
import numpy as np
from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.linear_operators import LinearOperator
from pygeoinf.gaussian_measure import GaussianMeasure
from pygeoinf.forward_problem import LinearForwardProblem
from pygeoinf.linear_solvers import CholeskySolver, LUSolver, ResidualTrackingCallback
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

    def test_least_squares_data_space_equivalence(
        self, forward_problem: LinearForwardProblem, synthetic_data: np.ndarray
    ):
        """
        Tests that the 'data_space' formalism yields the exact same model solution
        as the 'model_space' formalism for an underdetermined problem.
        """
        damping = 0.1
        solver = CholeskySolver(galerkin=True)

        # 1. Solve using the standard model space formulation
        lsq_model = LinearLeastSquaresInversion(
            forward_problem, formalism="model_space"
        )
        op_model = lsq_model.least_squares_operator(damping, solver)
        u_model = op_model(synthetic_data)

        # 2. Solve using the newly added data space formulation
        lsq_data = LinearLeastSquaresInversion(forward_problem, formalism="data_space")
        op_data = lsq_data.least_squares_operator(damping, solver)
        u_data = op_data(synthetic_data)

        # 3. Compare the components to ensure exact mathematical equivalence
        u_model_vec = forward_problem.model_space.to_components(u_model)
        u_data_vec = forward_problem.model_space.to_components(u_data)

        assert np.allclose(u_model_vec, u_data_vec, atol=1e-8, rtol=1e-8)

    def test_invalid_formalism_raises_error(
        self, forward_problem: LinearForwardProblem
    ):
        """
        Tests that providing an invalid formalism string raises the appropriate error.
        """
        with pytest.raises(
            ValueError, match="formalism must be either 'model_space' or 'data_space'"
        ):
            LinearLeastSquaresInversion(forward_problem, formalism="spectral_space")

    def test_woodbury_exact_equivalence(self, forward_problem: LinearForwardProblem):
        """
        Verifies that the Woodbury preconditioner exactly matches the inverse
        of the dense data-space least-squares normal operator.
        """
        damping = 0.5
        solver = LUSolver(galerkin=False)
        inversion = LinearLeastSquaresInversion(forward_problem)

        # 1. Compute the Woodbury preconditioner
        woodbury_precon = inversion.woodbury_data_preconditioner(damping, solver)

        # 2. Extract exact dense matrices
        A = forward_problem.forward_operator.matrix(dense=True)
        R = forward_problem.data_error_measure.covariance.matrix(dense=True)

        # 3. Calculate exact inverse of normal operator: (A A^T + damping * R)^-1
        exact_normal_matrix = A @ A.T + damping * R
        exact_inverse_matrix = np.linalg.inv(exact_normal_matrix)

        # 4. Extract dense matrix from the Woodbury operator
        woodbury_matrix = woodbury_precon.matrix(dense=True)

        # 5. Compare
        assert np.allclose(woodbury_matrix, exact_inverse_matrix, atol=1e-8)

    def test_surrogate_woodbury_chaining(self, forward_problem: LinearForwardProblem):
        """
        Verifies that the surrogate wrapper correctly chains the surrogate
        least-squares inversion and the Woodbury preconditioner extraction.
        """
        damping = 0.1
        solver = LUSolver(galerkin=False)
        inversion = LinearLeastSquaresInversion(forward_problem)

        # Create an arbitrary "alternate" forward operator (e.g., scaled by 0.5)
        alt_A = 0.5 * forward_problem.forward_operator

        # 1. Generate via the chained method
        chained_precon = inversion.surrogate_woodbury_data_preconditioner(
            damping, solver, alternate_forward_operator=alt_A
        )

        # 2. Generate manually
        manual_surrogate = inversion.surrogate_inversion(
            alternate_forward_operator=alt_A
        )
        manual_precon = manual_surrogate.woodbury_data_preconditioner(damping, solver)

        # 3. Compare matrices
        assert np.allclose(
            chained_precon.matrix(dense=True), manual_precon.matrix(dense=True)
        )

    def test_woodbury_requires_data_error(self, forward_problem: LinearForwardProblem):
        """
        Verifies that the Woodbury identity fails safely if no data error measure is set.
        """
        # Create an inversion without a data error measure
        fp_no_noise = LinearForwardProblem(forward_problem.forward_operator)
        inversion = LinearLeastSquaresInversion(fp_no_noise)
        solver = LUSolver(galerkin=False)

        with pytest.raises(ValueError, match="Data error measure must be set"):
            inversion.woodbury_data_preconditioner(0.1, solver)

    def test_normal_residual_callback(
        self, forward_problem: LinearForwardProblem, synthetic_data: np.ndarray
    ):
        """
        Verifies that the callback generator wires the correct operator and RHS
        into the ResidualTrackingCallback for least-squares.
        """
        damping = 0.2
        inversion = LinearLeastSquaresInversion(forward_problem)

        callback = inversion.normal_residual_callback(
            damping, synthetic_data, print_progress=False
        )

        assert isinstance(callback, ResidualTrackingCallback)

        # Verify the operator inside the callback is the correctly damped normal operator
        expected_normal_matrix = inversion.normal_operator(damping).matrix(dense=True)
        callback_normal_matrix = callback.operator.matrix(dense=True)

        assert np.allclose(callback_normal_matrix, expected_normal_matrix)

        # Verify the target 'y' vector in the callback is the correct normal RHS
        expected_rhs = inversion.normal_rhs(synthetic_data)
        assert np.allclose(
            forward_problem.data_space.to_components(callback.y),
            forward_problem.data_space.to_components(expected_rhs),
        )

    def test_parameterized_least_squares(self, forward_problem: LinearForwardProblem):
        """
        Tests that LinearLeastSquaresInversion correctly generates a
        parameterized surrogate using the base class method.
        """
        # 1. Setup a model-space inversion
        lsq = LinearLeastSquaresInversion(forward_problem, formalism="model_space")

        # 2. Define a parameterization mapping R^2 -> Model Space (R^5)
        param_space = EuclideanSpace(dim=2)
        param_matrix = np.random.randn(forward_problem.model_space.dim, param_space.dim)
        param_op = LinearOperator.from_matrix(
            param_space, forward_problem.model_space, param_matrix
        )

        # 3. Create the surrogate
        surrogate = lsq.parameterized_inversion(param_op, dense=True)

        # 4. Verify class type and properties
        assert isinstance(surrogate, LinearLeastSquaresInversion)
        assert surrogate.formalism == "model_space"
        assert surrogate.model_space == param_space

        # 5. Verify the 'dense' flag propagated correctly to the surrogate's operator
        from pygeoinf.linear_operators import DenseMatrixLinearOperator

        assert isinstance(
            surrogate.forward_problem.forward_operator, DenseMatrixLinearOperator
        )

    def test_parameterized_numerical_consistency(
        self, forward_problem: LinearForwardProblem
    ):
        """
        Verifies that solving the parameterized inversion is numerically
        consistent with the full inversion.
        """
        damping = 0.1
        solver = CholeskySolver(galerkin=True)

        # 1. Parameterization: m = [1, 1, 1, 1, 1] * alpha
        param_space = EuclideanSpace(dim=1)
        ones_vec = np.ones(forward_problem.model_space.dim)
        M = LinearOperator.from_vector(forward_problem.model_space, ones_vec).adjoint

        lsq_full = LinearLeastSquaresInversion(forward_problem)
        lsq_param = lsq_full.parameterized_inversion(M)

        # 2. Synthetic data from a known parameter alpha=2.0
        alpha_true = param_space.from_components(np.array([2.0]))
        u_true = M(alpha_true)
        data = forward_problem.forward_operator(u_true)  # Noise-free for consistency

        # 3. Solve parameterized problem
        op_param = lsq_param.least_squares_operator(damping, solver)
        alpha_sol = op_param(data)

        # Verify alpha is close to 2.0 (regularization will pull it slightly away)
        assert np.allclose(alpha_sol, alpha_true, atol=0.2)

    def test_woodbury_model_exact_equivalence(
        self, forward_problem: LinearForwardProblem
    ):
        """
        Verifies that the model-space Woodbury preconditioner exactly matches
        the inverse of the dense model-space least-squares normal operator.
        """
        damping = 0.5
        solver = LUSolver(galerkin=False)
        inversion = LinearLeastSquaresInversion(
            forward_problem, formalism="model_space"
        )

        # 1. Compute the model-space Woodbury preconditioner
        woodbury_precon = inversion.woodbury_model_preconditioner(damping, solver)

        # 2. Extract exact dense matrices
        A = forward_problem.forward_operator.matrix(dense=True)
        R = forward_problem.data_error_measure.covariance.matrix(dense=True)
        R_inv = np.linalg.inv(R)

        # 3. Calculate exact inverse of model-space normal operator: (A^T R^-1 A + damping * I)^-1
        exact_normal_matrix = A.T @ R_inv @ A + damping * np.eye(A.shape[1])
        exact_inverse_matrix = np.linalg.inv(exact_normal_matrix)

        # 4. Extract dense matrix from the Woodbury operator
        woodbury_matrix = woodbury_precon.matrix(dense=True)

        # 5. Compare
        assert np.allclose(woodbury_matrix, exact_inverse_matrix, atol=1e-8)

    def test_surrogate_woodbury_model_chaining(
        self, forward_problem: LinearForwardProblem
    ):
        """
        Verifies that the surrogate wrapper correctly chains the surrogate
        least-squares inversion and the model-space Woodbury preconditioner extraction.
        """
        damping = 0.1
        solver = LUSolver(galerkin=False)
        inversion = LinearLeastSquaresInversion(forward_problem)

        alt_A = 0.5 * forward_problem.forward_operator

        # 1. Generate via the chained method
        chained_precon = inversion.surrogate_woodbury_model_preconditioner(
            damping, solver, alternate_forward_operator=alt_A
        )

        # 2. Generate manually
        manual_surrogate = inversion.surrogate_inversion(
            alternate_forward_operator=alt_A
        )
        manual_precon = manual_surrogate.woodbury_model_preconditioner(damping, solver)

        # 3. Compare matrices
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
        inversion = LinearLeastSquaresInversion(fp_no_noise)
        solver = LUSolver(galerkin=False)

        with pytest.raises(ValueError, match="Data error measure must be set"):
            inversion.woodbury_model_preconditioner(0.1, solver)


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

    def test_constrained_least_squares_data_space_equivalence(self):
        """
        Tests that the 'data_space' formalism yields the exact same constrained model
        solution as the 'model_space' formalism.
        """
        domain = EuclideanSpace(5)
        fp = LinearForwardProblem(domain.identity_operator())

        # Constraint: Mean(u) = 2.0
        codomain = EuclideanSpace(1)
        ones = np.ones((1, 5)) / 5.0
        B = LinearOperator.from_matrix(domain, codomain, ones)
        w = codomain.from_components(np.array([2.0]))
        constraint = AffineSubspace.from_linear_equation(B, w)

        data = domain.from_components(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

        # INCREASED DAMPING: Makes the matrix well-conditioned so floating-point
        # noise doesn't trigger the strict 1e-8 absolute tolerance.
        damping = 0.1
        solver = CholeskySolver(galerkin=True)

        # Solve model space
        solver_model = ConstrainedLinearLeastSquaresInversion(
            fp, constraint, formalism="model_space"
        )
        u_model = solver_model.least_squares_operator(damping, solver)(data)

        # Solve data space
        solver_data = ConstrainedLinearLeastSquaresInversion(
            fp, constraint, formalism="data_space"
        )
        u_data = solver_data.least_squares_operator(damping, solver)(data)

        # Compare
        u_model_vec = domain.to_components(u_model)
        u_data_vec = domain.to_components(u_data)

        assert np.allclose(u_model_vec, u_data_vec, atol=1e-8, rtol=1e-8)

    def test_constrained_normal_residual_callback(self):
        """
        Verifies that the constrained callback correctly shifts the data by the
        constraint's affine base before passing it to the unconstrained tracking logic.
        """
        domain = EuclideanSpace(5)
        fp = LinearForwardProblem(
            domain.identity_operator(),
            data_error_measure=GaussianMeasure.from_standard_deviation(domain, 1.0),
        )

        codomain = EuclideanSpace(1)
        ones = np.ones((1, 5)) / 5.0
        B = LinearOperator.from_matrix(domain, codomain, ones)
        w = codomain.from_components(np.array([2.0]))
        constraint = AffineSubspace.from_linear_equation(B, w)

        inversion = ConstrainedLinearLeastSquaresInversion(fp, constraint)
        data = domain.from_components(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        damping = 0.1

        callback = inversion.normal_residual_callback(
            damping, data, print_progress=False
        )

        assert isinstance(callback, ResidualTrackingCallback)

        # The underlying RHS should be derived from the shifted data: data - A(u_base)
        u_base = constraint.projection_operator.translation_part
        data_offset = fp.forward_operator(u_base)
        shifted_data = domain.subtract(data, data_offset)

        # We reach inside the class solely for validation of the expected shift
        expected_rhs = inversion._unconstrained_inversion.normal_rhs(shifted_data)

        assert np.allclose(
            domain.to_components(callback.y), domain.to_components(expected_rhs)
        )

    def test_parameterized_constrained_least_squares(self):
        """
        Verifies the pullback of an explicit constraint into a parameter space.
        """
        domain = EuclideanSpace(5)
        fp = LinearForwardProblem(domain.identity_operator())

        # 1. Setup explicit constraint: Mean(u) = 10.0
        codomain = EuclideanSpace(1)
        B = LinearOperator.from_matrix(domain, codomain, np.ones((1, 5)) / 5.0)
        w = codomain.from_components(np.array([10.0]))
        constraint = AffineSubspace.from_linear_equation(B, w)

        inversion = ConstrainedLinearLeastSquaresInversion(fp, constraint)

        # 2. Parameterization: m = [1, 1, 1, 1, 1] * alpha
        # Note: This basis vector IS the constraint mode, so it can satisfy it.
        param_space = EuclideanSpace(1)
        M = LinearOperator.from_vector(domain, np.ones(5)).adjoint

        # 3. Create Surrogate
        surrogate = inversion.parameterized_inversion(M, dense=True)

        assert isinstance(surrogate, ConstrainedLinearLeastSquaresInversion)
        assert surrogate.model_space == param_space
        # Verify the new constraint is 1D: (B @ M) * alpha = 10.0
        assert surrogate._constraint.constraint_operator.domain == param_space
        assert surrogate._constraint.constraint_operator.codomain == codomain

    def test_parameterized_constrained_dimension_error(self):
        """Tests that parameterizing with too few degrees of freedom fails."""
        domain = EuclideanSpace(5)
        fp = LinearForwardProblem(domain.identity_operator())

        # 2 constraints in R^5
        B = LinearOperator.from_matrix(domain, EuclideanSpace(2), np.random.randn(2, 5))
        constraint = AffineSubspace.from_linear_equation(B, EuclideanSpace(2).zero)
        inversion = ConstrainedLinearLeastSquaresInversion(fp, constraint)

        # Parameter space of only 1 dimension (cannot satisfy 2 constraints)
        M = LinearOperator.from_matrix(EuclideanSpace(1), domain, np.random.randn(5, 1))

        with pytest.raises(ValueError, match="smaller than the number of constraints"):
            inversion.parameterized_inversion(M)


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

    def test_min_norm_data_space_equivalence(
        self, forward_problem: LinearForwardProblem, synthetic_data: np.ndarray
    ):
        """
        Tests that the 'data_space' formalism yields the exact same minimum norm solution
        and derivative as the 'model_space' formalism.
        """
        solver = CholeskySolver(galerkin=True)

        # 1. Model space formulation
        min_norm_model = LinearMinimumNormInversion(
            forward_problem, formalism="model_space"
        )
        op_model = min_norm_model.minimum_norm_operator(
            solver, significance_level=0.95, rtol=1e-10
        )
        u_model = op_model(synthetic_data)
        deriv_model = op_model.derivative(synthetic_data)

        # 2. Data space formulation
        min_norm_data = LinearMinimumNormInversion(
            forward_problem, formalism="data_space"
        )
        op_data = min_norm_data.minimum_norm_operator(
            solver, significance_level=0.95, rtol=1e-10
        )
        u_data = op_data(synthetic_data)
        deriv_data = op_data.derivative(synthetic_data)

        # 3. Verify equivalences
        u_model_vec = forward_problem.model_space.to_components(u_model)
        u_data_vec = forward_problem.model_space.to_components(u_data)
        assert np.allclose(u_model_vec, u_data_vec, atol=1e-6, rtol=1e-6)

        # Check derivative push-through identity equivalence
        v = forward_problem.data_space.random()
        du_model = forward_problem.model_space.to_components(deriv_model(v))
        du_data = forward_problem.model_space.to_components(deriv_data(v))
        assert np.allclose(du_model, du_data, atol=1e-6, rtol=1e-6)


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

        # We use a std dev of 2.0. If it were 1.0, the minimum possible
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

    def test_constrained_min_norm_data_space_equivalence(self):
        """
        Tests data-space equivalence for the constrained discrepancy principle search.
        """
        domain = EuclideanSpace(5)
        fp = LinearForwardProblem(
            domain.identity_operator(),
            data_error_measure=GaussianMeasure.from_standard_deviation(domain, 1.0),
        )

        # Constraint: Mean(u) = 10.0
        codomain = EuclideanSpace(1)
        ones = np.ones((1, 5)) / 5.0
        B = LinearOperator.from_matrix(domain, codomain, ones)
        w = codomain.from_components(np.array([10.0]))
        constraint = AffineSubspace.from_linear_equation(B, w)

        data = domain.from_components(np.array([15.0, 5.0, 20.0, 0.0, 10.0]))
        solver = CholeskySolver(galerkin=True)

        # Model space
        inv_model = ConstrainedLinearMinimumNormInversion(
            fp, constraint, formalism="model_space"
        )
        op_model = inv_model.minimum_norm_operator(
            solver, significance_level=0.95, rtol=1e-10
        )

        # Data space
        inv_data = ConstrainedLinearMinimumNormInversion(
            fp, constraint, formalism="data_space"
        )
        op_data = inv_data.minimum_norm_operator(
            solver, significance_level=0.95, rtol=1e-10
        )

        assert np.allclose(
            domain.to_components(op_model(data)),
            domain.to_components(op_data(data)),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_parameterized_constrained_geometric_fails(self):
        """
        Verifies that geometric-only constraints (no explicit B operator)
        cannot be parameterized.
        """
        domain = EuclideanSpace(5)
        fp = LinearForwardProblem(domain.identity_operator())

        # Create a purely geometric subspace (projection-based)
        # Tangent space is just the first 3 coordinates
        basis = [domain.basis_vector(i) for i in range(3)]

        # Suppress the expected UserWarning from subspaces.py
        with pytest.warns(
            UserWarning, match="Constructing a subspace from a tangent basis"
        ):
            constraint = AffineSubspace.from_tangent_basis(domain, basis)

        inversion = ConstrainedLinearMinimumNormInversion(fp, constraint)
        M = domain.identity_operator()

        with pytest.raises(NotImplementedError, match="explicit linear equation"):
            inversion.parameterized_inversion(M)


# =============================================================================
# Tests for Formalism Swapping API
# =============================================================================


class TestFormalismSwapping:
    """Tests the with_formalism and formalism override APIs for optimisation classes."""

    def test_unconstrained_with_formalism(self, forward_problem: LinearForwardProblem):
        lsq_data = LinearLeastSquaresInversion(forward_problem, formalism="data_space")
        lsq_model = lsq_data.with_formalism("model_space")

        assert lsq_model.formalism == "model_space"
        assert isinstance(lsq_model, LinearLeastSquaresInversion)
        assert lsq_data.formalism == "data_space"

        min_norm_data = LinearMinimumNormInversion(
            forward_problem, formalism="data_space"
        )
        min_norm_model = min_norm_data.with_formalism("model_space")

        assert min_norm_model.formalism == "model_space"
        assert isinstance(min_norm_model, LinearMinimumNormInversion)

    def test_constrained_with_formalism_and_override(
        self, forward_problem: LinearForwardProblem
    ):
        domain = forward_problem.model_space
        codomain = EuclideanSpace(1)
        B = LinearOperator.from_matrix(domain, codomain, np.ones((1, 5)))
        w = codomain.from_components(np.array([1.0]))
        constraint = AffineSubspace.from_linear_equation(B, w)

        # 1. Test with_formalism
        c_lsq_data = ConstrainedLinearLeastSquaresInversion(
            forward_problem, constraint, formalism="data_space"
        )
        c_lsq_model = c_lsq_data.with_formalism("model_space")

        assert c_lsq_model.formalism == "model_space"
        assert c_lsq_model._constraint is constraint

        # 2. Test parameterized formalism override
        param_space = EuclideanSpace(1)
        M = LinearOperator.from_vector(domain, np.ones(5)).adjoint

        surrogate = c_lsq_data.parameterized_inversion(M, formalism="model_space")

        assert surrogate.formalism == "model_space"
        assert surrogate.model_space == param_space


# =============================================================================
# Tests for Data Reduction Parameterization API
# =============================================================================


class TestDataReducedOptimisation:
    """
    Tests the data reduction surrogate generation across concrete optimisation classes.
    """

    def test_least_squares_data_reduction(self, forward_problem: LinearForwardProblem):
        """Tests data reduction for standard least-squares inversion."""
        lsq = LinearLeastSquaresInversion(forward_problem, formalism="model_space")

        reduced_space = EuclideanSpace(dim=1)
        reduction_matrix = np.random.randn(
            reduced_space.dim, forward_problem.data_space.dim
        )
        reduction_op = LinearOperator.from_matrix(
            forward_problem.data_space, reduced_space, reduction_matrix
        )

        surrogate = lsq.data_reduced_inversion(reduction_op, dense=True)

        assert isinstance(surrogate, LinearLeastSquaresInversion)
        assert surrogate.formalism == "model_space"
        assert surrogate.data_space == reduced_space

        from pygeoinf.linear_operators import DenseMatrixLinearOperator

        assert isinstance(
            surrogate.forward_problem.forward_operator, DenseMatrixLinearOperator
        )

    def test_constrained_least_squares_data_reduction(
        self, forward_problem: LinearForwardProblem
    ):
        """Tests that affine constraints are preserved during data space reduction."""
        domain = forward_problem.model_space
        codomain = EuclideanSpace(1)
        B = LinearOperator.from_matrix(domain, codomain, np.ones((1, domain.dim)))
        w = codomain.from_components(np.array([1.0]))
        constraint = AffineSubspace.from_linear_equation(B, w)

        inversion = ConstrainedLinearLeastSquaresInversion(forward_problem, constraint)

        reduced_space = EuclideanSpace(dim=1)
        S = LinearOperator.from_matrix(
            forward_problem.data_space,
            reduced_space,
            np.ones((1, forward_problem.data_space.dim)),
        )

        surrogate = inversion.data_reduced_inversion(S)

        assert isinstance(surrogate, ConstrainedLinearLeastSquaresInversion)
        assert surrogate.data_space == reduced_space
        # Verify the constraint object is passed through exactly as-is
        assert surrogate._constraint is constraint

    def test_minimum_norm_data_reduction(self, forward_problem: LinearForwardProblem):
        """Tests that Minimum Norm inversions inherit and construct data reductions properly."""
        inversion = LinearMinimumNormInversion(forward_problem, formalism="data_space")

        S = forward_problem.data_space.identity_operator()
        surrogate = inversion.data_reduced_inversion(S)

        assert isinstance(surrogate, LinearMinimumNormInversion)
        assert surrogate.formalism == "data_space"
        assert surrogate.data_space == forward_problem.data_space
