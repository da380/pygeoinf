"""Operators that know their own matrix, diagonals, or how to apply a block.

REVIEW2 §4.4: ``(A + B).matrix()``, ``A.adjoint.matrix()``, the Jacobi diagonal
of ``M + t I`` and a direct inverse's matrix were all re-derived by ``dim``
applications of operators that could have written the answer down. The hooks
``_known_matrix`` / ``_known_diagonals`` / ``apply_block`` let them. Every
check here runs on a dense, non-diagonal Gram matrix: the form bookkeeping in
an adjoint or a composition is exactly where a metric factor goes missing.
"""

from contextlib import contextmanager

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pygeoinf2.algebra.diagonal import DiagonalLinearOperator
from pygeoinf2.algebra.direct_sum import (
    BlockDiagonalLinearOperator,
    BlockLinearOperator,
    DirectSum,
)
from pygeoinf2.algebra.operators import LinearOperator, MatrixLinearOperator
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.numerics.preconditioners import JacobiPreconditioner
from pygeoinf2.numerics.randomised import (
    deflated_diagonal,
    random_diagonal,
    random_eig,
)
from pygeoinf2.numerics.solvers import CholeskySolver, EigenSolver, LUSolver
from pygeoinf2.probability.gaussian import GaussianMeasure
from pygeoinf2.testing import check_operator
from pygeoinf2.traits import Traits

from .conftest import make_dense_metric_space

OFFSETS = (-3, -1, 0, 1, 2)
FORMS = ("components", "galerkin")


def spd(space, rng):
    """A genuinely self-adjoint positive definite operator on a dense metric.

    Self-adjointness is symmetry of the *Galerkin* matrix, so ``G P G`` with
    ``P`` symmetric positive definite; its components matrix is ``P G``.
    """
    root = rng.normal(size=(space.dim, space.dim))
    P = root @ root.T + np.identity(space.dim)
    G = space.gram_matrix()
    return LinearOperator.from_matrix(
        space, space, G @ P @ G, form="galerkin"
    ).with_traits(Traits.POSITIVE_DEFINITE)


def probed(operator):
    """The same operator with every hook hidden, so ``matrix`` probes."""
    return LinearOperator.from_callables(
        operator.domain, operator.codomain, operator, adjoint=operator.adjoint
    )


@contextmanager
def counting_applications():
    """Count the leaf applications of matrix-backed and diagonal operators."""
    counts = {"n": 0}
    originals = {}
    for cls in (MatrixLinearOperator, DiagonalLinearOperator):
        original = cls._value

        def counting(self, x, _original=original):
            counts["n"] += 1
            return _original(self, x)

        originals[cls] = original
        cls._value = counting
    try:
        yield counts
    finally:
        for cls, original in originals.items():
            cls._value = original


@pytest.fixture
def family(rng):
    """A family of expressions over one dense-metric space."""
    space = make_dense_metric_space(5)
    other = make_dense_metric_space(4)
    A = LinearOperator.from_matrix(
        space, space, rng.normal(size=(5, 5)), form="components"
    )
    B = LinearOperator.from_matrix(
        space, space, rng.normal(size=(5, 5)), form="galerkin"
    )
    S = spd(space, rng)
    D = DiagonalLinearOperator(space, np.array([1.0, 2.0, -0.5, 3.0, 0.25]))
    R = LinearOperator.from_matrix(
        space, other, rng.normal(size=(4, 5)), form="components"
    )
    I = LinearOperator.identity(space)
    Z = LinearOperator.zero(space)
    expressions = {
        "A": A,
        "B (galerkin stored)": B,
        "D": D,
        "I": I,
        "Z": Z,
        "A + B": A + B,
        "2.5 A": 2.5 * A,
        "A @ B": A @ B,
        "A.adjoint": A.adjoint,
        "(A @ B).adjoint": (A @ B).adjoint,
        "A + 0.5 I": A + 0.5 * I,
        "D @ A": D @ A,
        "S + 0.5 I": S + 0.5 * I,
        "R (rectangular)": R,
        "R.adjoint": R.adjoint,
        "R @ A": R @ A,
    }
    return space, expressions


class TestKnownMatrix:
    def test_every_expression_reads_what_probing_would_find(self, family):
        space, expressions = family
        for name, operator in expressions.items():
            for form in FORMS:
                assert operator._known_matrix(form) is not None, (name, form)
                assert operator.matrix(form=form) == pytest.approx(
                    probed(operator).matrix(form=form)
                ), (name, form)

    def test_reading_applies_nothing(self, family):
        space, expressions = family
        with counting_applications() as counts:
            for operator in expressions.values():
                operator.matrix(form="components")
                operator.matrix(form="galerkin")
        assert counts["n"] == 0

    def test_a_callable_operator_still_probes(self, family):
        space, expressions = family
        wrapped = probed(expressions["A"])
        assert wrapped._known_matrix("components") is None
        assert wrapped.matrix() == pytest.approx(expressions["A"].matrix())

    def test_a_direct_inverse_is_read_from_its_factors(self, family, rng):
        space, expressions = family
        A, S = expressions["A"], expressions["S + 0.5 I"]
        for inverse in (LUSolver()(A), CholeskySolver()(S), EigenSolver()(S)):
            for form in FORMS:
                assert inverse._known_matrix(form) is not None
                assert inverse.matrix(form=form) == pytest.approx(
                    probed(inverse).matrix(form=form)
                )
                assert inverse.adjoint.matrix(form=form) == pytest.approx(
                    probed(inverse.adjoint).matrix(form=form)
                )
            check_operator(inverse, rng=rng)

    def test_an_lu_solver_factorises_once(self, family, monkeypatch, rng):
        """The fix for the adjoint reuse factorised *twice*: O(n^3), doubled,
        for a solver whose point is to do it once (REVIEW2 3.5)."""
        import pygeoinf2.numerics.solvers as solvers

        calls = {"n": 0}
        original = solvers.lu_factor

        def counting(matrix, *args, **kwargs):
            calls["n"] += 1
            return original(matrix, *args, **kwargs)

        monkeypatch.setattr(solvers, "lu_factor", counting)
        space, expressions = family
        inverse = LUSolver()(expressions["A"])
        x = space.random(rng=rng)
        inverse(x)
        inverse.adjoint(x)
        inverse.matrix()
        assert calls["n"] == 1

    def test_the_solver_takes_workers_for_the_probe(self, family):
        space, expressions = family
        wrapped = probed(expressions["S + 0.5 I"]).with_traits(
            Traits.POSITIVE_DEFINITE
        )
        serial = CholeskySolver()(wrapped).matrix()
        parallel = CholeskySolver(n_jobs=2)(wrapped).matrix()
        assert serial == pytest.approx(parallel)

    def test_a_scaled_identity_keeps_its_traits_on_any_metric(self, family):
        """``sigma I`` is self-adjoint and definite in every metric; the
        diagonal operator it folds to could not deduce that from its values on
        a dense one, and ``(sigma I)(sigma I)`` -- every isotropic covariance --
        came out claiming nothing."""
        space, expressions = family
        scaled = 0.5 * expressions["I"]
        assert Traits.POSITIVE_DEFINITE & scaled.traits
        assert Traits.POSITIVE_DEFINITE & (scaled @ scaled).traits
        assert Traits.POSITIVE_DEFINITE & (scaled + scaled).traits
        assert not (Traits.POSITIVE_SEMIDEFINITE & (-2.0 * scaled).traits)
        measure = GaussianMeasure.from_standard_deviation(space, 1.5)
        assert Traits.POSITIVE_DEFINITE & measure.covariance.traits


class TestKnownDiagonals:
    def test_every_expression_reads_what_probing_would_find(self, family):
        space, expressions = family
        for name, operator in expressions.items():
            for form in FORMS:
                assert operator.diagonals(offsets=OFFSETS, form=form) == pytest.approx(
                    probed(operator).diagonals(offsets=OFFSETS, form=form)
                ), (name, form)

    def test_the_main_diagonal_of_a_sum_with_the_identity_is_read(self, family):
        """The Jacobi case on a dense metric: ``M + t I`` needs the Gram
        diagonal, which the space supplies, and not a probe of ``M``."""
        space, expressions = family
        with counting_applications() as counts:
            expressions["S + 0.5 I"].diagonals(offsets=(0,), form="galerkin")
            expressions["A + B"].diagonals(offsets=OFFSETS, form="galerkin")
            expressions["A.adjoint"].diagonals(offsets=OFFSETS, form="galerkin")
            JacobiPreconditioner()(expressions["S + 0.5 I"])
        assert counts["n"] == 0

    def test_a_composition_still_probes_and_may_do_so_in_parallel(self, family):
        space, expressions = family
        product = expressions["A @ B"]
        assert product._known_diagonals((0,), "galerkin") is None
        serial = product.diagonals(offsets=OFFSETS, form="galerkin")
        parallel = product.diagonals(offsets=OFFSETS, form="galerkin", n_jobs=2)
        assert serial == pytest.approx(parallel)

    def test_a_sparse_matrix_is_read_without_densifying(self, rng):
        space = make_dense_metric_space(6)
        dense = np.triu(rng.normal(size=(6, 6)), -1)
        operator = LinearOperator.from_matrix(
            space, space, csr_matrix(dense), form="components"
        )
        assert operator.diagonals(offsets=OFFSETS, form="components") == pytest.approx(
            probed(operator).diagonals(offsets=OFFSETS, form="components")
        )
        assert operator.diagonals(offsets=OFFSETS, form="galerkin") == pytest.approx(
            probed(operator).diagonals(offsets=OFFSETS, form="galerkin")
        )


class TestApplyBlock:
    def test_every_expression_agrees_with_one_at_a_time(self, family, rng):
        space, expressions = family
        for name, operator in expressions.items():
            vectors = [operator.domain.random(rng=rng) for _ in range(3)]
            block = operator.apply_block(vectors)
            for x, y in zip(vectors, block):
                assert np.allclose(y, operator(x)), name
            images = [operator.codomain.random(rng=rng) for _ in range(3)]
            for y, x in zip(images, operator.adjoint.apply_block(images)):
                assert np.allclose(x, operator.adjoint(y)), name

    def test_a_column_operator_applies_as_a_block(self, family, rng):
        space, _ = family
        vectors = [space.random(rng=rng) for _ in range(3)]
        operator = LinearOperator.from_vectors(space, vectors)
        coefficients = [rng.normal(size=3) for _ in range(4)]
        for c, y in zip(coefficients, operator.apply_block(coefficients)):
            assert np.allclose(y, operator(c))
        images = [space.random(rng=rng) for _ in range(2)]
        for y, c in zip(images, operator.adjoint.apply_block(images)):
            assert np.allclose(c, operator.adjoint(y))

    def test_the_general_route_runs_in_parallel(self, family, rng):
        space, expressions = family
        wrapped = probed(expressions["A @ B"])
        vectors = [space.random(rng=rng) for _ in range(4)]
        serial = wrapped.apply_block(vectors)
        parallel = wrapped.apply_block(vectors, n_jobs=2)
        for a, b in zip(serial, parallel):
            assert np.allclose(a, b)

    def test_the_randomised_routines_do_not_move_with_the_job_count(
        self, family, rng
    ):
        space, expressions = family
        S = probed(expressions["S + 0.5 I"]).with_traits(Traits.POSITIVE_DEFINITE)
        serial = random_eig(S, rank=3, rng=np.random.default_rng(1))
        parallel = random_eig(S, rank=3, rng=np.random.default_rng(1), n_jobs=2)
        assert serial.eigenvalues == pytest.approx(parallel.eigenvalues)
        assert random_diagonal(
            S, samples=6, rng=np.random.default_rng(2)
        ) == pytest.approx(
            random_diagonal(S, samples=6, rng=np.random.default_rng(2), n_jobs=2)
        )
        assert deflated_diagonal(
            S, rank=2, samples=6, rng=np.random.default_rng(3)
        ) == pytest.approx(
            deflated_diagonal(
                S, rank=2, samples=6, rng=np.random.default_rng(3), n_jobs=2
            )
        )

    def test_sample_expectation_takes_workers(self, family):
        space, _ = family
        measure = GaussianMeasure.from_standard_deviation(space, 1.5)
        serial = measure.sample_expectation(8, rng=np.random.default_rng(4))
        parallel = measure.sample_expectation(
            8, rng=np.random.default_rng(4), n_jobs=2
        )
        assert np.allclose(serial, parallel)


class TestBlockOperators:
    @pytest.fixture
    def blocks(self, rng):
        first = make_dense_metric_space(3)
        second = make_dense_metric_space(2)
        A = LinearOperator.from_matrix(
            first, first, rng.normal(size=(3, 3)), form="galerkin"
        )
        B = LinearOperator.from_matrix(
            second, first, rng.normal(size=(3, 2)), form="components"
        )
        C = LinearOperator.from_matrix(
            first, second, rng.normal(size=(2, 3)), form="components"
        )
        D = DiagonalLinearOperator(second, np.array([2.0, -1.0]))
        return first, second, A, B, C, D

    def test_a_block_operator_reads_its_matrix_and_diagonal(self, blocks, rng):
        first, second, A, B, C, D = blocks
        operator = BlockLinearOperator([[A, B], [C, D]])
        for form in FORMS:
            assert operator.matrix(form=form) == pytest.approx(
                probed(operator).matrix(form=form)
            )
            assert operator.diagonals(offsets=(0,), form=form) == pytest.approx(
                probed(operator).diagonals(offsets=(0,), form=form)
            )
        assert operator._known_diagonals((1,), "galerkin") is None
        check_operator(operator, rng=rng)

    def test_a_block_diagonal_operator_reads_and_applies_blocks(self, blocks, rng):
        first, second, A, B, C, D = blocks
        operator = BlockDiagonalLinearOperator([A, D])
        for form in FORMS:
            assert operator.matrix(form=form) == pytest.approx(
                probed(operator).matrix(form=form)
            )
            assert operator.diagonals(offsets=(0,), form=form) == pytest.approx(
                probed(operator).diagonals(offsets=(0,), form=form)
            )
        vectors = [operator.domain.random(rng=rng) for _ in range(3)]
        for x, y in zip(vectors, operator.apply_block(vectors)):
            for part, expected in zip(y, operator(x)):
                assert np.allclose(part, expected)

    def test_a_block_with_a_probed_entry_falls_back(self, blocks):
        first, second, A, B, C, D = blocks
        operator = BlockLinearOperator([[A, B], [probed(C), D]])
        assert operator._known_matrix("components") is None
        assert operator.matrix() == pytest.approx(probed(operator).matrix())


class CountingSpace(type(make_dense_metric_space(3))):
    """A dense-metric space that counts its coordinate conversions."""

    def __init__(self, gram):
        super().__init__(gram)
        self.analyses = 0
        self.syntheses = 0

    def to_components(self, x):
        self.analyses += 1
        return super().to_components(x)

    def from_components(self, c):
        self.syntheses += 1
        return super().from_components(c)

    def reset(self):
        self.analyses = self.syntheses = 0


class TestComponentsAction:
    """Products and sums of operators that act on components stay in
    components: one conversion in, one out, whatever the length of the run."""

    @pytest.fixture
    def counted(self, rng):
        model = CountingSpace(make_dense_metric_space(6).gram_matrix())
        data = CountingSpace(make_dense_metric_space(4).gram_matrix())
        A = LinearOperator.from_matrix(
            model, data, rng.normal(size=(4, 6)), form="components"
        )
        return model, data, A, spd(model, rng), spd(data, rng)

    def test_the_normal_operator_converts_once_each_way(self, counted, rng):
        model, data, A, Q, R = counted
        normal = A @ Q @ A.adjoint + R
        y = data.random(rng=rng)
        expected = data.add(A(Q(A.adjoint(y))), R(y))
        model.reset(), data.reset()
        result = normal(y)
        assert np.allclose(result, expected)
        assert (data.analyses, data.syntheses) == (1, 1)
        assert (model.analyses, model.syntheses) == (0, 0)

    def test_every_expression_agrees_when_chained(self, family, rng):
        space, expressions = family
        for name, operator in expressions.items():
            action = operator._components_action()
            assert action is not None, name
            x = operator.domain.random(rng=rng)
            fused = operator.codomain.from_components(
                action(operator.domain.to_components(x))
            )
            assert np.allclose(fused, probed(operator)(x)), name
            adjoint = operator._components_adjoint_action()
            assert adjoint is not None, name
            y = operator.codomain.random(rng=rng)
            fused = operator.domain.from_components(
                adjoint(operator.codomain.to_components(y))
            )
            assert np.allclose(fused, probed(operator).adjoint(y)), name

    def test_a_run_stops_at_an_operator_without_the_action(self, counted, rng):
        """``Q A* N^-1 A Q`` with an opaque middle: two conversions on each
        side of it, not two per factor."""
        model, data, A, Q, R = counted
        middle = probed(R)
        product = Q @ A.adjoint @ middle @ A @ Q
        x = model.random(rng=rng)
        expected = Q(A.adjoint(middle(A(Q(x)))))
        model.reset(), data.reset()
        result = product(x)
        assert np.allclose(result, expected)
        # One conversion into the run ``A Q`` and one out of ``Q A*`` on the
        # model side; on the data side one out of the first run and one into
        # the second, plus the pair the opaque middle spends on its own --
        # against five of each per side when every factor converted.
        assert (model.analyses, model.syntheses) == (1, 1)
        assert (data.analyses, data.syntheses) == (2, 2)
        model.reset(), data.reset()
        pulled = product.adjoint(x)
        assert np.allclose(pulled, Q(A.adjoint(middle.adjoint(A(Q(x))))))
        assert (model.analyses, model.syntheses) == (1, 1)
        assert (data.analyses, data.syntheses) == (2, 2)

    def test_a_direct_inverse_acts_on_components(self, counted, rng):
        model, data, A, Q, R = counted
        normal = A @ Q @ A.adjoint + R
        normal = normal.with_traits(Traits.POSITIVE_DEFINITE)
        for solver in (CholeskySolver(), LUSolver()):
            inverse = solver(normal)
            gain = Q @ A.adjoint @ inverse
            y = data.random(rng=rng)
            expected = Q(A.adjoint(probed(inverse)(y)))
            model.reset(), data.reset()
            result = gain(y)
            assert np.allclose(result, expected)
            assert (data.analyses, model.syntheses) == (1, 1)
            check_operator(gain, rng=rng)
