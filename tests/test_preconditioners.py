import pytest
import numpy as np
from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.linear_operators import DenseMatrixLinearOperator, LinearOperator
from pygeoinf.linear_solvers import CGSolver, FCGSolver, CholeskySolver
from pygeoinf.preconditioners import (
    IdentityPreconditioningMethod,
    JacobiPreconditioningMethod,
    SpectralPreconditioningMethod,
    IterativePreconditioningMethod,
)
from pygeoinf.linear_optimisation import LinearLeastSquaresInversion
from pygeoinf.forward_problem import LinearForwardProblem

# --- Fixtures ---


@pytest.fixture
def space():
    """Returns a 10-dimensional Euclidean space for testing."""
    return EuclideanSpace(10)


@pytest.fixture
def forward_operator(space):
    """
    Returns a random forward operator A.
    The normal operator will be constructed as (A* A + alpha I).
    """
    np.random.seed(42)
    return DenseMatrixLinearOperator(space, space, np.random.randn(10, 10))


@pytest.fixture
def x(space):
    """Returns a random model vector x_true."""
    return space.random()


@pytest.fixture
def inversion_setup(forward_operator):
    """Initializes the inversion setup for the forward operator."""
    fwd = LinearForwardProblem(forward_operator)
    return LinearLeastSquaresInversion(fwd)


def get_reference_solution(inversion, damping, data):
    """
    Computes the 'true' regularized solution using a direct solver.
    This provides the exact numerical target for iterative solvers.
    """
    direct_solver = CholeskySolver()
    # least_squares_operator using a direct solver yields the exact bias-point
    ls_op = inversion.least_squares_operator(damping, direct_solver)
    return ls_op(data)


# --- Tests ---


def test_identity_preconditioner_logic(inversion_setup):
    """Verifies that IdentityPreconditioningMethod acts as a no-op."""
    inversion = inversion_setup
    normal_op = inversion.normal_operator(damping=0.1)

    method = IdentityPreconditioningMethod()
    precond = method(normal_op)

    test_vec = normal_op.domain.random()
    # The identity preconditioner should return the vector unchanged
    assert np.allclose(precond(test_vec), test_vec)


def test_jacobi_variable_convergence(inversion_setup):
    """Tests the stochastic Jacobi estimator using variable-rank sampling."""
    inversion = inversion_setup
    normal_op = inversion.normal_operator(damping=0.5)

    method = JacobiPreconditioningMethod(num_samples=20, method="variable", rtol=1e-1)
    precond = method(normal_op)

    # Check space compatibility instead of shape
    assert precond.domain == normal_op.domain
    assert precond.codomain == normal_op.codomain

    # Ensure diagonal values are strictly positive for the damped system
    diag = precond.matrix(dense=True, galerkin=True).diagonal()
    assert np.all(diag > 0)


def test_spectral_preconditioner_in_least_squares(inversion_setup, x):
    """Verifies solver converges to the correct regularized solution."""
    inversion = inversion_setup
    damping = 0.01
    data = inversion.forward_problem.forward_operator(x)

    # Target is the damped solution, not x itself
    x_ref = get_reference_solution(inversion, damping, data)

    spec_method = SpectralPreconditioningMethod(damping=damping, rank=5)
    solver = CGSolver(rtol=1e-10)

    ls_op = inversion.least_squares_operator(
        damping, solver, preconditioner=spec_method
    )
    x_sol = ls_op(data)

    assert np.allclose(x_sol, x_ref, rtol=1e-7, atol=1e-7)


def test_iterative_preconditioner_with_fcg(inversion_setup, x):
    """Verifies FCG converges to the correct regularized solution."""
    inversion = inversion_setup
    damping = 0.1
    data = inversion.forward_problem.forward_operator(x)

    x_ref = get_reference_solution(inversion, damping, data)

    inner_cg = CGSolver(maxiter=3)
    iter_method = IterativePreconditioningMethod(inner_cg)
    outer_solver = FCGSolver(rtol=1e-10)

    ls_op = inversion.least_squares_operator(
        damping, outer_solver, preconditioner=iter_method
    )
    x_sol = ls_op(data)

    assert np.allclose(x_sol, x_ref, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize(
    "method_class",
    [
        IdentityPreconditioningMethod,
        lambda: JacobiPreconditioningMethod(num_samples=10),
    ],
)
def test_preconditioner_api_consistency(inversion_setup, method_class):
    """Ensures all methods properly resolve to a LinearOperator via the factory."""
    inversion = inversion_setup
    normal_op = inversion.normal_operator(damping=1.0)

    method = method_class()
    precond = method(normal_op)

    assert isinstance(precond, LinearOperator)
