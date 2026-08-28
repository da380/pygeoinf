"""Randomised linear algebra: range finding, factorisation, estimators."""

import numpy as np
import pytest

from pygeoinf2.algebra.operators import LinearOperator
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.numerics.randomised import (
    Estimate,
    LowRankCholesky,
    LowRankEig,
    LowRankSVD,
    random_cholesky,
    random_diagonal,
    random_eig,
    random_range,
    random_svd,
    random_trace,
)
from pygeoinf2.testing import check_operator, check_traits
from pygeoinf2.traits import Traits

from .conftest import make_weighted_space
from .doubles import NoCoordinatesError, Opaque, OpaqueSpace, StrictSpace

N, RANK = 40, 6


@pytest.fixture
def exact_low_rank(rng):
    """A positive semidefinite operator of known, exact rank."""
    X = EuclideanSpace(N)
    root = rng.normal(size=(N, RANK))
    matrix = root @ root.T
    operator = LinearOperator.from_matrix(
        X, X, matrix, traits=Traits.POSITIVE_SEMIDEFINITE, form="components"
    )
    return X, operator, matrix


class TestRangeFinding:
    def test_a_fixed_rank_basis_is_orthonormal(self, exact_low_rank, rng):
        X, A, _ = exact_low_rank
        basis = random_range(A, rank=RANK, rng=rng)
        for i, u in enumerate(basis):
            for j, v in enumerate(basis):
                assert X.inner_product(u, v) == pytest.approx(
                    1.0 if i == j else 0.0, abs=1e-10
                )

    def test_it_captures_the_range(self, exact_low_rank, rng):
        """Projecting onto the basis should leave nothing behind."""
        X, A, matrix = exact_low_rank
        basis = random_range(A, rank=RANK, rng=rng)
        x = X.random(rng=rng)
        image = A(x)
        residual = X.copy(image)
        for q in basis:
            residual = X.axpy(-X.inner_product(residual, q), q, residual)
        assert X.norm(residual) < 1e-8 * X.norm(image)

    def test_the_adaptive_mode_finds_the_true_rank(self, exact_low_rank, rng):
        _, A, _ = exact_low_rank
        basis = random_range(A, rng=rng, rtol=1e-8, block_size=4)
        assert len(basis) == RANK

    def test_the_adaptive_mode_respects_its_ceiling(self, exact_low_rank, rng):
        _, A, _ = exact_low_rank
        assert len(random_range(A, rng=rng, rtol=1e-12, max_rank=3)) <= 3

    def test_power_iteration_helps_a_slow_spectrum(self, rng):
        """Where subspace iteration earns its keep."""
        X = EuclideanSpace(60)
        values = 1.0 / np.arange(1, 61) ** 0.5  # slowly decaying
        basis = np.linalg.qr(rng.normal(size=(60, 60)))[0]
        matrix = basis @ np.diag(values) @ basis.T
        A = LinearOperator.from_matrix(
            X, X, matrix, traits=Traits.POSITIVE_DEFINITE, form="components"
        )

        def captured(power):
            q = random_range(A, rank=8, oversampling=2, power=power, rng=rng)
            projector = np.column_stack([X.to_components(v) for v in q])
            return np.linalg.norm(projector.T @ matrix, "fro")

        assert captured(2) > captured(0)

    def test_a_rank_of_zero_is_refused(self, exact_low_rank, rng):
        _, A, _ = exact_low_rank
        with pytest.raises(ValueError, match="positive"):
            random_range(A, rank=0, rng=rng)


class TestFactorisations:
    def test_eig_recovers_an_exactly_low_rank_operator(self, exact_low_rank, rng):
        X, A, matrix = exact_low_rank
        decomposition = random_eig(A, rank=RANK, rng=rng)
        assert isinstance(decomposition, LowRankEig)
        assert decomposition.rank == RANK

        expected = np.sort(np.linalg.eigvalsh(matrix))[::-1][:RANK]
        assert np.allclose(
            np.sort(decomposition.eigenvalues)[::-1], expected, atol=1e-8
        )

        x = X.random(rng=rng)
        assert np.allclose(decomposition(x), matrix @ x, atol=1e-8)

    def test_eig_carries_the_right_traits(self, exact_low_rank, rng):
        _, A, _ = exact_low_rank
        decomposition = random_eig(A, rank=RANK, rng=rng)
        assert Traits.SELF_ADJOINT & decomposition.traits
        assert Traits.POSITIVE_SEMIDEFINITE & decomposition.traits
        check_operator(decomposition, rng=rng)
        check_traits(decomposition, rng=rng)

    def test_eig_is_refused_for_a_non_self_adjoint_operator(self, rng):
        X = EuclideanSpace(10)
        A = LinearOperator.from_matrix(
            X, X, rng.normal(size=(10, 10)), form="components"
        )
        with pytest.raises(ValueError, match="self-adjoint"):
            random_eig(A, rank=3, rng=rng)

    def test_svd_recovers_a_rectangular_operator(self, rng):
        X, Y = EuclideanSpace(30), EuclideanSpace(20)
        matrix = rng.normal(size=(20, RANK)) @ rng.normal(size=(RANK, 30))
        A = LinearOperator.from_matrix(X, Y, matrix, form="components")

        decomposition = random_svd(A, rank=RANK, rng=rng)
        assert isinstance(decomposition, LowRankSVD)
        expected = np.linalg.svd(matrix, compute_uv=False)[:RANK]
        assert np.allclose(decomposition.singular_values, expected, atol=1e-8)

        x = X.random(rng=rng)
        assert np.allclose(decomposition(x), matrix @ x, atol=1e-8)
        check_operator(decomposition, rng=rng)

    def test_svd_singular_values_are_ordered(self, rng):
        X, Y = EuclideanSpace(30), EuclideanSpace(20)
        matrix = rng.normal(size=(20, RANK)) @ rng.normal(size=(RANK, 30))
        A = LinearOperator.from_matrix(X, Y, matrix, form="components")
        values = random_svd(A, rank=RANK, rng=rng).singular_values
        assert np.all(np.diff(values) <= 1e-12)

    def test_cholesky_gives_a_usable_covariance_factor(self, exact_low_rank, rng):
        X, A, matrix = exact_low_rank
        factorisation = random_cholesky(A, rank=RANK, rng=rng)
        assert isinstance(factorisation, LowRankCholesky)
        assert Traits.POSITIVE_SEMIDEFINITE & factorisation.traits

        x = X.random(rng=rng)
        assert np.allclose(factorisation(x), matrix @ x, atol=1e-8)

        # The factor is exactly what a Gaussian needs to sample from.
        from pygeoinf2.probability import GaussianMeasure

        measure = GaussianMeasure(X, covariance_factor=factorisation.factor)
        assert measure.can_sample
        assert np.allclose(measure.covariance(x), matrix @ x, atol=1e-8)

    def test_cholesky_is_refused_for_an_indefinite_operator(self, rng):
        X = EuclideanSpace(10)
        matrix = rng.normal(size=(10, 10))
        A = LinearOperator.from_matrix(
            X, X, matrix + matrix.T, traits=Traits.SELF_ADJOINT, form="components"
        )
        with pytest.raises(ValueError, match="positive semidefinite"):
            random_cholesky(A, rank=3, rng=rng)

    def test_the_eig_factor_is_an_isometry(self, exact_low_rank, rng):
        """Which is what makes U D U* recognisable as semidefinite."""
        _, A, _ = exact_low_rank
        factor = random_eig(A, rank=RANK, rng=rng).factor
        assert Traits.ISOMETRY & factor.traits
        check_traits(factor, rng=rng)

    def test_apply_function_acts_on_the_retained_spectrum(self, exact_low_rank, rng):
        X, A, matrix = exact_low_rank
        decomposition = random_eig(A, rank=RANK, rng=rng)
        root = decomposition.apply_function(np.sqrt)
        x = X.random(rng=rng)
        assert np.allclose(root(root(x)), decomposition(x), atol=1e-8)


class TestEstimators:
    def test_trace_matches_on_an_orthonormal_space(self, exact_low_rank, rng):
        _, A, matrix = exact_low_rank
        estimate = random_trace(A, samples=6000, rng=rng)
        assert isinstance(estimate, Estimate)
        assert abs(estimate.value - np.trace(matrix)) < 4.0 * estimate.standard_error

    def test_trace_is_right_on_a_weighted_space(self, rng):
        """Where v1's probe distribution gives a different number entirely.

        The trace of an operator is basis-independent, and equals the trace of
        its component matrix. Hutchinson recovers it only when the probes have
        identity covariance *on the space*; standard normal components give
        ``tr(G A_c)`` instead.
        """
        X = make_weighted_space()
        values = np.array([1.0, 2.0, 3.0, 4.0])
        A = LinearOperator.self_adjoint(
            X,
            lambda x: X.from_components(values * X.to_components(x)),
            traits=Traits.SELF_ADJOINT,
        )
        exact = float(values.sum())
        wrong = float((X.metric_values * values).sum())
        assert not np.isclose(exact, wrong)  # the two really do differ here

        estimate = random_trace(A, samples=8000, rng=rng)
        assert abs(estimate.value - exact) < 4.0 * estimate.standard_error
        assert abs(estimate.value - wrong) > 4.0 * estimate.standard_error

    def test_the_standard_error_shrinks(self, exact_low_rank, rng):
        _, A, _ = exact_low_rank
        few = random_trace(A, samples=200, rng=rng)
        many = random_trace(A, samples=5000, rng=rng)
        assert many.standard_error < few.standard_error

    def test_trace_needs_an_endomorphism(self, rng):
        X, Y = EuclideanSpace(5), EuclideanSpace(3)
        A = LinearOperator.from_matrix(X, Y, rng.normal(size=(3, 5)), form="components")
        with pytest.raises(ValueError, match="space to itself"):
            random_trace(A, rng=rng)

    def test_diagonal_of_the_component_matrix(self, rng):
        X = EuclideanSpace(12)
        matrix = np.diag(np.arange(1.0, 13.0)) + 0.05 * rng.normal(size=(12, 12))
        A = LinearOperator.from_matrix(X, X, matrix, form="components")
        estimate = random_diagonal(A, samples=4000, form="components", rng=rng)
        assert np.allclose(estimate, np.diag(matrix), rtol=0.15)

    def test_diagonal_of_the_galerkin_matrix(self, rng):
        X = make_weighted_space()
        values = np.array([1.0, 2.0, 3.0, 4.0])
        A = LinearOperator.self_adjoint(
            X,
            lambda x: X.from_components(values * X.to_components(x)),
            traits=Traits.SELF_ADJOINT,
        )
        estimate = random_diagonal(A, samples=2000, form="galerkin", rng=rng)
        assert np.allclose(estimate, X.metric_values * values, rtol=1e-8)

    def test_an_unknown_form_is_refused(self, exact_low_rank, rng):
        _, A, _ = exact_low_rank
        with pytest.raises(ValueError, match="Unknown form"):
            random_diagonal(A, form="nonsense", rng=rng)

    def test_the_diagonal_needs_coordinates(self, rng):
        strict = StrictSpace(make_weighted_space())
        A = LinearOperator.self_adjoint(strict, lambda x: x, traits=Traits.SELF_ADJOINT)
        with pytest.raises((TypeError, NoCoordinatesError)):
            random_diagonal(A, rng=rng)


class TestCoordinateFreedom:
    """Everything but the diagonal estimator runs without a component map.

    ``OpaqueSpace`` is used rather than ``StrictSpace`` here, and it is the
    stronger test: it is not a ``CoordinateSpace`` at all, so a component map
    does not merely raise, it does not exist. ``StrictSpace`` cannot serve,
    because drawing white noise on a coordinate space is *legitimately* a
    coordinate operation -- ``G^(-1/2)`` is a statement about a basis -- and a
    genuinely coordinate-free space supplies its own draw instead, as this one
    does.
    """

    @pytest.fixture
    def opaque_problem(self, rng):
        space = OpaqueSpace(np.array([1.0, 4.0, 9.0, 0.25]))
        values = np.array([4.0, 3.0, 0.0, 0.0])  # rank two

        def action(x):
            return Opaque(values * x.data)

        A = LinearOperator.self_adjoint(
            space, action, traits=Traits.POSITIVE_SEMIDEFINITE
        )
        return space, A, values

    def test_the_space_really_has_no_coordinates(self, opaque_problem):
        from pygeoinf2.algebra.spaces import CoordinateSpace

        space, _, _ = opaque_problem
        assert not isinstance(space, CoordinateSpace)
        assert not hasattr(space, "to_components")

    def test_range_finding_is_coordinate_free(self, opaque_problem, rng):
        _, A, _ = opaque_problem
        assert len(random_range(A, rng=rng, rtol=1e-8, block_size=2)) == 2

    def test_eig_is_coordinate_free(self, opaque_problem, rng):
        _, A, _ = opaque_problem
        decomposition = random_eig(A, rank=2, rng=rng)
        assert np.allclose(np.sort(decomposition.eigenvalues), [3.0, 4.0], atol=1e-8)

    def test_svd_is_coordinate_free(self, opaque_problem, rng):
        _, A, _ = opaque_problem
        decomposition = random_svd(A, rank=2, rng=rng)
        assert np.allclose(decomposition.singular_values, [4.0, 3.0], atol=1e-8)

    def test_cholesky_is_coordinate_free(self, opaque_problem, rng):
        space, A, _ = opaque_problem
        factorisation = random_cholesky(A, rank=2, rng=rng)
        x = space.random(rng=rng)
        assert space.norm(space.subtract(factorisation(x), A(x))) < 1e-8

    def test_trace_is_coordinate_free(self, opaque_problem, rng):
        _, A, values = opaque_problem
        estimate = random_trace(A, samples=4000, rng=rng)
        assert abs(estimate.value - values.sum()) < 4.0 * estimate.standard_error
