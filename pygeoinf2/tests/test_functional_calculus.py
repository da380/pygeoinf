"""Functional calculus: Lanczos, the diagonal fast path, and trait gating."""

import numpy as np
import pytest
import scipy.linalg as sla

from pygeoinf2.algebra.diagonal import DiagonalLinearOperator
from pygeoinf2.algebra.operators import LinearOperator
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.numerics.functional_calculus import (
    OperatorFunction,
    apply_operator_function,
    iter_lanczos_tridiagonalise,
    lanczos_tridiagonalise,
    operator_exp,
    operator_function,
    operator_inverse_sqrt,
    operator_log,
    operator_power,
    operator_quadratic_form,
    operator_sqrt,
)
from pygeoinf2.testing import check_operator, check_traits
from pygeoinf2.traits import Traits

from .conftest import make_weighted_space
from .doubles import NoCoordinatesError, OpaqueSpace, StrictSpace

N = 16


def spd(rng, n=N, shift=None):
    root = rng.normal(size=(n, n))
    return root @ root.T + (n if shift is None else shift) * np.identity(n)


@pytest.fixture
def problem(rng):
    X = EuclideanSpace(N)
    matrix = spd(rng)
    A = LinearOperator.from_component_matrix(
        X, X, matrix, traits=Traits.POSITIVE_DEFINITE
    )
    return X, A, matrix


class TestLanczos:
    def test_the_basis_is_orthonormal(self, problem, rng):
        X, A, _ = problem
        basis, _ = lanczos_tridiagonalise(A, X.random(rng=rng), 8)
        for i, u in enumerate(basis):
            for j, v in enumerate(basis):
                assert X.inner_product(u, v) == pytest.approx(
                    1.0 if i == j else 0.0, abs=1e-10
                )

    def test_the_matrix_is_the_projected_operator(self, problem, rng):
        """Q* A Q == T, which is what makes the whole method work."""
        X, A, _ = problem
        basis, matrix = lanczos_tridiagonalise(A, X.random(rng=rng), 6)
        projected = np.array([[X.inner_product(A(v), u) for v in basis] for u in basis])
        assert np.allclose(projected, matrix, atol=1e-9)

    def test_it_is_tridiagonal(self, problem, rng):
        X, A, _ = problem
        _, matrix = lanczos_tridiagonalise(A, X.random(rng=rng), 8)
        assert np.allclose(np.triu(matrix, 2), 0.0)
        assert np.allclose(np.tril(matrix, -2), 0.0)
        assert np.allclose(matrix, matrix.T)

    def test_a_full_run_recovers_the_spectrum(self, problem, rng):
        X, A, matrix = problem
        _, tridiagonal = lanczos_tridiagonalise(A, X.random(rng=rng), N)
        assert np.allclose(
            np.sort(np.linalg.eigvalsh(tridiagonal)),
            np.sort(np.linalg.eigvalsh(matrix)),
            rtol=1e-8,
        )

    def test_reorthogonalisation_matters(self, problem, rng):
        """Without it, Lanczos loses orthogonality after very few steps."""
        X, A, _ = problem
        start = X.random(rng=rng)
        with_reorth, _ = lanczos_tridiagonalise(A, start, N, reorthogonalise=True)
        without, _ = lanczos_tridiagonalise(A, start, N, reorthogonalise=False)

        def worst_overlap(basis):
            return max(
                abs(X.inner_product(basis[i], basis[j]))
                for i in range(len(basis))
                for j in range(i + 1, len(basis))
            )

        assert worst_overlap(with_reorth) < 1e-10
        assert worst_overlap(without) > worst_overlap(with_reorth)

    def test_it_yields_progressively(self, problem, rng):
        X, A, _ = problem
        sizes = [
            matrix.shape[0]
            for _, matrix in iter_lanczos_tridiagonalise(A, X.random(rng=rng), 5)
        ]
        assert sizes == [1, 2, 3, 4, 5]

    def test_a_zero_start_is_refused(self, problem):
        X, A, _ = problem
        with pytest.raises(ValueError, match="nonzero"):
            lanczos_tridiagonalise(A, X.zero(), 3)

    def test_an_invariant_subspace_terminates_early(self, rng):
        """Exact termination, not failure."""
        X = EuclideanSpace(4)
        A = DiagonalLinearOperator(X, np.array([2.0, 2.0, 5.0, 7.0]))
        basis, _ = lanczos_tridiagonalise(A, X.basis_vector(0), 4)
        assert len(basis) == 1  # e_0 is already an eigenvector


class TestAgainstDenseMatrixFunctions:
    @pytest.mark.parametrize("name", ["sqrt", "log", "exp"])
    def test_apply_matches_scipy(self, name, problem, rng):
        X, A, matrix = problem
        if name == "exp":
            matrix = matrix / 50.0
            A = LinearOperator.from_component_matrix(
                X, X, matrix, traits=Traits.POSITIVE_DEFINITE
            )
        function, reference = {
            "sqrt": (np.sqrt, sla.sqrtm),
            "log": (np.log, sla.logm),
            "exp": (np.exp, sla.expm),
        }[name]
        x = X.random(rng=rng)
        got = apply_operator_function(A, function, x, max_iterations=N)
        expected = np.real(reference(matrix)) @ x
        assert np.allclose(got, expected, rtol=1e-8)

    def test_quadratic_form_matches(self, problem, rng):
        X, A, matrix = problem
        x = X.random(rng=rng)
        got = operator_quadratic_form(A, np.log, x, max_iterations=N)
        expected = x @ np.real(sla.logm(matrix)) @ x
        assert got == pytest.approx(expected, rel=1e-8)

    def test_the_square_root_squares_back(self, problem, rng):
        X, A, matrix = problem
        root = operator_sqrt(A)
        x = X.random(rng=rng)
        assert np.allclose(root(root(x)), matrix @ x, rtol=1e-6)

    def test_the_inverse_square_root(self, problem, rng):
        X, A, matrix = problem
        inverse_root = operator_inverse_sqrt(A)
        x = X.random(rng=rng)
        assert np.allclose(inverse_root(inverse_root(A(x))), x, rtol=1e-6)

    def test_a_fractional_power(self, problem, rng):
        X, A, matrix = problem
        x = X.random(rng=rng)
        got = operator_power(A, 0.25)(x)
        expected = np.real(sla.fractional_matrix_power(matrix, 0.25)) @ x
        assert np.allclose(got, expected, rtol=1e-7)


class TestOperatorFunction:
    def test_it_is_a_self_adjoint_operator(self, problem, rng):
        _, A, _ = problem
        root = operator_sqrt(A)
        assert isinstance(root, OperatorFunction)
        assert Traits.SELF_ADJOINT & root.traits
        check_operator(root, rng=rng)
        check_traits(root, rng=rng)

    def test_the_result_carries_the_right_claim(self, problem):
        _, A, _ = problem
        assert Traits.POSITIVE_SEMIDEFINITE & operator_sqrt(A).traits
        assert Traits.POSITIVE_DEFINITE & operator_exp(A).traits
        # A logarithm is indefinite, and does not pretend otherwise.
        assert not (Traits.POSITIVE_SEMIDEFINITE & operator_log(A).traits)

    def test_it_composes_into_the_algebra(self, problem, rng):
        X, A, matrix = problem
        root = operator_sqrt(A)
        product = root @ root
        x = X.random(rng=rng)
        assert np.allclose(product(x), matrix @ x, rtol=1e-6)


class TestTraitGating:
    def test_a_non_self_adjoint_operator_is_refused(self, rng):
        X = EuclideanSpace(N)
        A = LinearOperator.from_component_matrix(X, X, rng.normal(size=(N, N)))
        with pytest.raises(ValueError, match="self-adjoint"):
            operator_function(A, np.sqrt)

    def test_a_square_root_needs_semidefiniteness(self, rng):
        X = EuclideanSpace(N)
        matrix = spd(rng)
        A = LinearOperator.from_component_matrix(
            X, X, matrix, traits=Traits.SELF_ADJOINT
        )
        with pytest.raises(ValueError, match="square root requires"):
            operator_sqrt(A)

    def test_a_logarithm_needs_definiteness(self, rng):
        X = EuclideanSpace(N)
        A = LinearOperator.from_component_matrix(
            X, X, spd(rng), traits=Traits.POSITIVE_SEMIDEFINITE
        )
        with pytest.raises(ValueError, match="logarithm requires"):
            operator_log(A)

    def test_the_message_points_at_check_traits(self, rng):
        X = EuclideanSpace(N)
        A = LinearOperator.from_component_matrix(X, X, rng.normal(size=(N, N)))
        with pytest.raises(ValueError, match="check_traits"):
            operator_function(A, np.sqrt)


class TestDiagonalFastPath:
    def test_a_diagonal_operator_is_evaluated_exactly(self, rng):
        """No Krylov iteration: f is applied to the eigenvalues."""
        X = make_weighted_space()
        values = np.array([1.0, 4.0, 9.0, 16.0])
        A = DiagonalLinearOperator(X, values)

        root = operator_function(A, np.sqrt)
        assert isinstance(root, DiagonalLinearOperator)
        assert np.allclose(root.eigenvalues, np.sqrt(values))

    def test_the_two_paths_agree(self, rng):
        """The dispatch is an optimisation, not a different answer."""
        X = EuclideanSpace(6)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        diagonal = DiagonalLinearOperator(X, values)
        dense = LinearOperator.from_component_matrix(
            X, X, np.diag(values), traits=Traits.POSITIVE_DEFINITE
        )
        x = X.random(rng=rng)
        assert np.allclose(
            operator_sqrt(diagonal)(x),
            apply_operator_function(dense, np.sqrt, x, max_iterations=6),
            rtol=1e-9,
        )

    def test_log_determinant_is_exact(self, rng):
        X = EuclideanSpace(5)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        A = DiagonalLinearOperator(X, values)
        assert A.log_determinant == pytest.approx(float(np.sum(np.log(values))))


class TestDiagonalOperator:
    def test_traits_are_read_off_the_spectrum(self, rng):
        X = make_weighted_space()
        positive = DiagonalLinearOperator(X, np.array([1.0, 2.0, 3.0, 4.0]))
        assert Traits.POSITIVE_DEFINITE & positive.traits
        check_traits(positive, rng=rng)

        semidefinite = DiagonalLinearOperator(X, np.array([0.0, 2.0, 3.0, 4.0]))
        assert Traits.POSITIVE_SEMIDEFINITE & semidefinite.traits
        assert not (Traits.POSITIVE_DEFINITE & semidefinite.traits)

        indefinite = DiagonalLinearOperator(X, np.array([-1.0, 2.0, 3.0, 4.0]))
        assert Traits.SELF_ADJOINT & indefinite.traits
        assert not (Traits.POSITIVE_SEMIDEFINITE & indefinite.traits)
        check_traits(indefinite, rng=rng)

    def test_a_projection_is_recognised(self, rng):
        X = EuclideanSpace(4)
        A = DiagonalLinearOperator(X, np.array([1.0, 1.0, 0.0, 0.0]))
        assert Traits.IDEMPOTENT & A.traits
        check_traits(A, rng=rng)

    def test_symmetry_needs_a_diagonal_metric(self, rng):
        """Diagonal in the components is not enough on a general Gram matrix."""
        from .conftest import make_dense_metric_space

        X = make_dense_metric_space()
        assert not X.has_diagonal_metric
        A = DiagonalLinearOperator(X, np.array([1.0, 2.0, 3.0]))
        assert not (Traits.SELF_ADJOINT & A.traits)
        check_operator(A, rng=rng)  # the adjoint is still correct

    def test_the_algebra_stays_diagonal(self, rng):
        X = make_weighted_space()
        A = DiagonalLinearOperator(X, np.array([1.0, 2.0, 3.0, 4.0]))
        B = DiagonalLinearOperator(X, np.array([0.5, 0.5, 0.5, 0.5]))

        assert isinstance(A + B, DiagonalLinearOperator)
        assert isinstance(A @ B, DiagonalLinearOperator)
        assert isinstance(3.0 * A, DiagonalLinearOperator)
        assert np.allclose((A + B).eigenvalues, [1.5, 2.5, 3.5, 4.5])
        assert np.allclose((A @ B).eigenvalues, [0.5, 1.0, 1.5, 2.0])

    def test_it_needs_a_basis(self):
        X = OpaqueSpace(np.array([1.0, 2.0]))
        with pytest.raises(TypeError, match="no coordinate map"):
            DiagonalLinearOperator(X, np.array([1.0, 2.0]))

    def test_the_calculus_is_gated(self, rng):
        X = make_weighted_space()
        indefinite = DiagonalLinearOperator(X, np.array([-1.0, 2.0, 3.0, 4.0]))
        with pytest.raises(ValueError, match="square root"):
            indefinite.sqrt
        singular = DiagonalLinearOperator(X, np.array([0.0, 2.0, 3.0, 4.0]))
        with pytest.raises(ZeroDivisionError, match="singular"):
            singular.inverse


class TestCoordinateFreedom:
    def test_lanczos_never_touches_components(self, rng):
        """The whole point: f(A) on a space with no component map."""
        base = make_weighted_space()
        strict = StrictSpace(base)
        values = np.array([1.0, 2.0, 3.0, 4.0])
        A = LinearOperator.self_adjoint(
            strict,
            lambda x: base.from_components(values * base.to_components(x)),
            traits=Traits.POSITIVE_DEFINITE,
        )
        x = strict.random(rng=rng)
        root = apply_operator_function(A, np.sqrt, x, max_iterations=4)
        assert strict.norm(strict.subtract(A(x), root)) > 0.0  # it did something
        # Applying it twice recovers A x.
        again = apply_operator_function(A, np.sqrt, root, max_iterations=4)
        assert strict.norm(strict.subtract(again, A(x))) < 1e-8 * strict.norm(A(x))

    def test_the_diagonal_path_does_need_coordinates(self, rng):
        """The negative control: eigenvalues are a statement about a basis."""
        strict = StrictSpace(make_weighted_space())
        A = DiagonalLinearOperator(strict, np.array([1.0, 2.0, 3.0, 4.0]))
        with pytest.raises(NoCoordinatesError):
            A(strict.random(rng=rng))


class TestLogDeterminant:
    """``log det A == tr(log A)``, densely and by stochastic Lanczos.

    The two routes share no code: one forms the matrix and factorises it, the
    other never forms anything and reaches the answer through a Krylov
    iteration and Hutchinson's estimator. They must agree, and on a space whose
    metric is not the identity they must agree about *which* determinant — the
    component matrix's, since ``det(G A_c)`` is a property of the metric as
    much as of the operator.
    """

    @pytest.fixture(params=["euclidean", "weighted", "dense-metric"])
    def operator(self, request, rng):
        from .conftest import make_dense_metric_space, make_weighted_space

        space = {
            "euclidean": lambda: EuclideanSpace(24),
            "weighted": make_weighted_space,
            "dense-metric": make_dense_metric_space,
        }[request.param]()
        root = rng.normal(size=(space.dim, space.dim))
        return LinearOperator.from_derivative_matrix(
            space,
            space,
            root @ root.T + space.dim * np.identity(space.dim),
            traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE,
        )

    def test_the_dense_route_is_the_operators_own_determinant(self, operator):
        from pygeoinf2.numerics.functional_calculus import log_determinant

        expected = float(np.linalg.slogdet(operator.matrix(form="components"))[1])
        estimate = log_determinant(operator, method="dense")
        assert estimate.value == pytest.approx(expected, abs=1e-9)
        assert estimate.standard_error == 0.0

    @pytest.mark.slow
    def test_the_stochastic_route_agrees_within_its_error(self, operator):
        """A stochastic estimate without its error is uninterpretable, so the
        test is written in units of that error rather than in a fixed
        tolerance: four standard errors is a real statement about a Hutchinson
        estimator, and 1e-6 would not be."""
        from pygeoinf2.numerics.functional_calculus import log_determinant

        exact = log_determinant(operator, method="dense").value
        estimate = log_determinant(
            operator,
            method="stochastic",
            samples=4000,
            rng=np.random.default_rng(1),
            max_iterations=60,
            rtol=1e-10,
        )
        assert estimate.standard_error > 0.0
        assert abs(estimate.value - exact) < 4.0 * estimate.standard_error

    @pytest.mark.slow
    def test_auto_goes_dense_only_when_it_can_afford_to(self, operator):
        from pygeoinf2.numerics.functional_calculus import log_determinant

        exact = log_determinant(operator, method="dense")
        assert log_determinant(operator, method="auto").standard_error == 0.0
        stochastic = log_determinant(
            operator,
            method="auto",
            dense_limit=1,
            samples=2000,
            rng=np.random.default_rng(2),
            max_iterations=60,
            rtol=1e-10,
        )
        assert stochastic.standard_error > 0.0
        assert abs(stochastic.value - exact.value) < 4.0 * stochastic.standard_error

    def test_it_refuses_what_it_cannot_do(self, operator, rng):
        from pygeoinf2.numerics.functional_calculus import log_determinant

        with pytest.raises(ValueError, match="'auto', 'dense' or 'stochastic'"):
            log_determinant(operator, method="lanczos")
        space = operator.domain
        indefinite = LinearOperator.from_derivative_matrix(
            space,
            space,
            -np.identity(space.dim),
            traits=Traits.SELF_ADJOINT,
        )
        with pytest.raises(ValueError, match="POSITIVE_DEFINITE"):
            log_determinant(indefinite, method="dense")
