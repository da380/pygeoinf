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
from pygeoinf2.numerics.functional_calculus import log_determinant
from pygeoinf2.testing import check_operator, check_traits
from pygeoinf2.traits import Traits

from .conftest import DenseMetricSpace, make_weighted_space
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


def dense_metric(n, rng):
    """A space whose Gram matrix is dense, which is the only kind that tells a
    metric-correct method from a metric-naive one.

    Kept well conditioned on purpose: the point of the fixture is that the
    metric is *not diagonal*, and a Gram condition number in the hundred
    thousands would only test floating point.
    """
    root = np.eye(n) + 0.15 * np.tril(rng.standard_normal((n, n)), -1)
    return DenseMetricSpace(root @ root.T)


class TestAdaptiveRangeIsIncremental:
    """The adaptive range finder rebuilt its whole basis every round, redoing
    every earlier vector's orthogonalisation and throwing away the residuals it
    had just computed to test convergence."""

    @staticmethod
    def low_rank(space, true_rank, rng):
        basis, _ = np.linalg.qr(rng.standard_normal((space.dim, space.dim)))
        eigenvalues = np.concatenate(
            [np.linspace(10.0, 1.0, true_rank), np.zeros(space.dim - true_rank)]
        )
        matrix = basis @ np.diag(eigenvalues) @ basis.T
        return matrix, LinearOperator.self_adjoint(
            space, lambda c: matrix @ c, traits=Traits.SELF_ADJOINT
        )

    def test_it_finds_the_range_with_far_less_work(self, rng):
        """Measured at dim 600 and rank 120: 33749 inner products against
        82788, and the same rank at the same 8e-16 projection error."""
        space = EuclideanSpace(600)
        matrix, operator = self.low_rank(space, 120, rng)

        counted = {"n": 0}
        original = type(space).inner_product
        type(space).inner_product = lambda self, a, b: (
            counted.__setitem__("n", counted["n"] + 1),
            original(self, a, b),
        )[1]
        try:
            basis = random_range(
                operator,
                rtol=1e-8,
                block_size=10,
                max_rank=200,
                rng=np.random.default_rng(1),
            )
        finally:
            type(space).inner_product = original

        assert len(basis) == 120
        stacked = np.array(basis)
        residual = matrix - stacked.T @ (stacked @ matrix)
        assert np.linalg.norm(residual) < 1e-10 * np.linalg.norm(matrix)
        assert counted["n"] < 60000  # the rebuilding route needed 82788

    def test_the_basis_is_still_orthonormal(self, rng):
        """Which is the thing the rebuilding was there to guarantee."""
        space = EuclideanSpace(200)
        _, operator = self.low_rank(space, 40, rng)
        basis = random_range(
            operator, rtol=1e-8, block_size=8, max_rank=100, rng=np.random.default_rng(2)
        )
        stacked = np.array(basis)
        assert stacked @ stacked.T == pytest.approx(np.eye(len(basis)), abs=1e-10)


class TestRandomisedOnADenseMetric:
    """The review found these tested only on diagonal Gram matrices, which
    cannot tell a metric-correct implementation from a metric-naive one."""

    def test_random_diagonal_reads_the_galerkin_diagonal(self, rng):
        space = dense_metric(30, rng)
        values = rng.uniform(1.0, 5.0, 30)
        operator = LinearOperator.from_matrix(
            space,
            space,
            np.diag(values),
            form="components",
            traits=Traits.SELF_ADJOINT,
        )

        estimate = random_diagonal(
            operator, samples=4000, form="galerkin", rng=np.random.default_rng(3)
        )
        exact = np.diag(space.apply_gram(np.diag(values)))
        assert estimate == pytest.approx(exact, rel=0.2)

    def test_random_trace_is_the_trace(self, rng):
        space = dense_metric(20, rng)
        matrix = rng.standard_normal((20, 20))
        matrix = matrix @ matrix.T
        operator = LinearOperator.from_matrix(
            space, space, matrix, form="galerkin", traits=Traits.SELF_ADJOINT
        )

        estimate = random_trace(operator, samples=6000, rng=np.random.default_rng(4))
        exact = np.trace(space.solve_gram(matrix))
        assert estimate.value == pytest.approx(exact, rel=0.15)

    def test_random_eig_recovers_the_operator(self, rng):
        space = dense_metric(24, rng)
        matrix = rng.standard_normal((24, 6))
        matrix = matrix @ matrix.T
        operator = LinearOperator.from_matrix(
            space,
            space,
            matrix,
            form="galerkin",
            traits=Traits.SELF_ADJOINT | Traits.POSITIVE_SEMIDEFINITE,
        )

        decomposition = random_eig(operator, rank=6, rng=np.random.default_rng(5))
        check_operator(decomposition, rng=rng)
        for _ in range(5):
            vector = space.random(rng=rng)
            assert space.norm(
                space.subtract(decomposition(vector), operator(vector))
            ) < 1e-8 * space.norm(operator(vector))

    def test_random_svd_recovers_the_operator(self, rng):
        space = dense_metric(20, rng)
        other = EuclideanSpace(12)
        matrix = rng.standard_normal((12, 20))
        operator = LinearOperator.from_matrix(space, other, matrix, form="galerkin")

        decomposition = random_svd(operator, rank=12, rng=np.random.default_rng(6))
        check_operator(decomposition, rng=rng)
        for _ in range(5):
            vector = space.random(rng=rng)
            assert other.norm(
                other.subtract(decomposition(vector), operator(vector))
            ) < 1e-8 * max(other.norm(operator(vector)), 1e-30)


class TestSamplingToATolerance:
    """A stochastic estimator that must be told a sample count is asking the
    caller to guess the answer's accuracy in advance. The Estimate type already
    carries the standard error; these stop when it is small enough."""

    @staticmethod
    def operator(n, rng):
        matrix = rng.standard_normal((n, n))
        matrix = matrix @ matrix.T / n + np.diag(rng.uniform(2.0, 6.0, n))
        space = EuclideanSpace(n)
        return matrix, LinearOperator.self_adjoint(
            space, lambda c: matrix @ c, traits=Traits.POSITIVE_DEFINITE
        )

    def test_a_tighter_trace_tolerance_draws_more(self, rng):
        matrix, operator = self.operator(200, rng)

        loose = random_trace(
            operator, samples=20, rtol=5e-2, rng=np.random.default_rng(1)
        )
        tight = random_trace(
            operator, samples=20, rtol=1e-2, rng=np.random.default_rng(1)
        )
        assert tight.samples > loose.samples
        assert tight.standard_error < loose.standard_error
        assert tight.standard_error <= 1e-2 * abs(tight.value)

    def test_the_diagonal_tolerance_is_the_accuracy_it_delivers(self, rng):
        """The stopping statistic is ``||standard error|| / ||estimate||``,
        which was chosen because it predicts the achieved error: asking for
        1e-2 gets 1.06e-2, and 2e-2 gets 1.95e-2. The worst entry's own
        standard error does not predict it -- the maximum over many entries
        runs several standard errors out, and a rule built on it stopped at the
        first block whatever tolerance it was given."""
        matrix, operator = self.operator(120, rng)
        exact = np.diag(matrix)

        achieved = {}
        for tolerance in (4e-2, 1e-2):
            estimate = random_diagonal(
                operator, samples=20, rtol=tolerance, rng=np.random.default_rng(1)
            )
            achieved[tolerance] = np.linalg.norm(estimate - exact) / np.linalg.norm(
                exact
            )
        assert achieved[1e-2] < achieved[4e-2]
        assert achieved[1e-2] < 2e-2

    def test_the_ceiling_is_honoured(self, rng):
        """A tolerance that cannot be met must stop somewhere, and say how
        many it drew."""
        _, operator = self.operator(60, rng)
        estimate = random_trace(
            operator,
            samples=10,
            rtol=1e-12,
            max_samples=50,
            rng=np.random.default_rng(2),
        )
        assert estimate.samples == 50

    def test_a_nonsense_tolerance_is_refused(self, rng):
        _, operator = self.operator(20, rng)
        with pytest.raises(ValueError, match="tolerance"):
            random_trace(operator, rtol=0.0)
        with pytest.raises(ValueError, match="tolerance"):
            random_diagonal(operator, rtol=1.5)

    def test_the_log_determinant_passes_it_down(self, rng):
        """Its own ``rtol`` is the *inner* Lanczos budget; the sampling
        tolerance is a separate one, and tightening the wrong one buys
        nothing."""
        matrix, operator = self.operator(80, rng)

        loose = log_determinant(
            operator, method="stochastic", samples=20, rng=np.random.default_rng(3)
        )
        tight = log_determinant(
            operator,
            method="stochastic",
            samples=20,
            sample_rtol=2e-3,
            rng=np.random.default_rng(3),
        )
        assert tight.samples > loose.samples
        assert tight.standard_error < loose.standard_error


class TestComponentFastPaths:
    """The randomised routines do their arithmetic on component arrays when
    the space has them, converting each vector once each way.

    On a spectral space an inner product analyses both arguments, so the
    coordinate-free route paid O(k^2) transforms for O(k) vectors: measured at
    lmax 64, ``random_eig(rank=20)`` spent 5794 analyses on 120 operator
    applications. The count is asserted here so the regression would be loud.
    """

    @staticmethod
    def counted_transforms(monkeypatch):
        import pyshtools.expand as expand

        counts = {"analysis": 0, "synthesis": 0}
        analysis, synthesis = expand.SHExpandDH, expand.MakeGridDH

        def count_analysis(*args, **kwargs):
            counts["analysis"] += 1
            return analysis(*args, **kwargs)

        def count_synthesis(*args, **kwargs):
            counts["synthesis"] += 1
            return synthesis(*args, **kwargs)

        monkeypatch.setattr(expand, "SHExpandDH", count_analysis)
        monkeypatch.setattr(expand, "MakeGridDH", count_synthesis)
        return counts

    def test_the_sphere_pays_one_analysis_per_vector(self, monkeypatch):
        from pygeoinf2.symmetric_space.sphere import Sobolev

        pytest.importorskip("pyshtools")
        space = Sobolev(16, 2.0, 0.3)
        # An operator whose own cost is exactly two transforms per application.
        values = 1.0 / (1.0 + np.arange(space.dim))
        operator = LinearOperator.self_adjoint(
            space,
            lambda x: space.from_components(values * space.to_components(x)),
            traits=Traits.POSITIVE_DEFINITE,
        )
        rank, oversampling, power = 8, 4, 1
        counts = self.counted_transforms(monkeypatch)
        decomposition = random_eig(
            operator,
            rank=rank,
            oversampling=oversampling,
            power=power,
            rng=np.random.default_rng(7),
        )
        probes = rank + oversampling
        applications = probes * (1 + 2 * power + 1)  # probes, power steps, images
        # Every analysis is either the operator's own or the single one the
        # fast paths make of each application's result: never a per-pair
        # analysis. The coordinate-free route measured 5794 for 120
        # applications at lmax 64.
        assert counts["analysis"] <= 2 * applications
        assert counts["synthesis"] <= 2 * applications + 2 * probes
        # One power step on a 1/(1+i) spectrum: approximate, as the method is.
        assert np.allclose(decomposition.eigenvalues, np.sort(values)[::-1][:rank], rtol=0.05)

    def test_the_two_routes_agree_on_a_dense_metric(self, rng):
        """The component route against the coordinate-free one it replaces."""
        from pygeoinf2.algebra.spaces import CoordinateSpace

        space = dense_metric(30, rng)
        matrix = rng.standard_normal((30, 8))
        operator = LinearOperator.from_matrix(
            space,
            space,
            matrix @ matrix.T,
            form="galerkin",
            traits=Traits.SELF_ADJOINT | Traits.POSITIVE_SEMIDEFINITE,
        )
        fast = random_eig(operator, rank=8, rng=np.random.default_rng(9))
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            CoordinateSpace, "uses_component_fast_paths", property(lambda self: False)
        )
        try:
            slow = random_eig(operator, rank=8, rng=np.random.default_rng(9))
        finally:
            monkeypatch.undo()
        assert fast.eigenvalues == pytest.approx(slow.eigenvalues, rel=1e-10)
        for _ in range(3):
            vector = space.random(rng=rng)
            assert space.norm(space.subtract(fast(vector), slow(vector))) < 1e-9

    def test_the_factor_is_an_isometry_in_the_metric(self, rng):
        space = dense_metric(25, rng)
        matrix = rng.standard_normal((25, 25))
        operator = LinearOperator.from_matrix(
            space, space, matrix @ matrix.T, form="galerkin", traits=Traits.SELF_ADJOINT
        )
        decomposition = random_eig(operator, rank=6, rng=np.random.default_rng(10))
        columns = decomposition.factor.columns
        gram = columns.T @ space.apply_gram_to_columns(columns)
        assert gram == pytest.approx(np.eye(6), abs=1e-10)
        check_traits(decomposition.factor, rng=rng)

    def test_the_strict_space_takes_the_coordinate_free_route(self, rng):
        """A space that forbids its coordinate map still gets an answer."""
        base = make_weighted_space()
        strict = StrictSpace(base)
        values = np.array([4.0, 3.0, 0.0, 0.0])
        operator = LinearOperator.self_adjoint(
            strict,
            lambda x: base.from_components(values * base.to_components(x)),
            traits=Traits.POSITIVE_SEMIDEFINITE,
        )
        basis = strict.orthonormal_basis([strict.random(rng=rng) for _ in range(3)])
        assert len(basis) == 3
        gram = np.array([[strict.inner_product(a, b) for b in basis] for a in basis])
        assert gram == pytest.approx(np.eye(3), abs=1e-10)
        # White noise on a coordinate space is legitimately a coordinate
        # operation, so the range finder is fed probes drawn on the base.
        probes = [operator(base.white_noise(rng=rng)) for _ in range(3)]
        assert len(strict.orthonormal_basis(probes)) == 2
