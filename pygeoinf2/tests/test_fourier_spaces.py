"""Periodic boxes: the coordinate map, the metric, and what follows from them."""

import numpy as np
import pytest

from pygeoinf2 import LinearOperator, Traits
from pygeoinf2.algebra.diagonal import DiagonalLinearOperator
from pygeoinf2.symmetric_space.base import SymmetricSpace
from pygeoinf2.symmetric_space import (
    Lebesgue,
    PeriodicBox,
    Sobolev,
    lift_formal_adjoint,
)
from pygeoinf2.testing import (
    check_coordinates,
    check_measure,
    check_operator,
    check_space,
    check_traits,
    check_white_noise,
)

SHAPES = [(8,), (7,), (16,), (6, 4), (5, 5), (4, 4, 4), (5, 4, 3)]


class TestCoordinateMap:
    """The packing of ``rfftn`` output is the one delicate part."""

    @pytest.mark.parametrize("shape", SHAPES)
    def test_the_dimension_is_the_number_of_grid_points(self, shape):
        """A real field has exactly that many degrees of freedom."""
        assert Lebesgue(shape).dim == int(np.prod(shape))

    @pytest.mark.parametrize("shape", SHAPES)
    def test_the_round_trip_is_exact(self, shape, rng):
        space = Lebesgue(shape)
        x = rng.normal(size=shape)
        assert np.allclose(space.from_components(space.to_components(x)), x, atol=1e-12)

    @pytest.mark.parametrize("shape", SHAPES)
    def test_parseval_against_the_grid_quadrature(self, shape, rng):
        """The Lebesgue basis is orthonormal, so the norm is the coefficient norm."""
        space = Lebesgue(shape)
        x = rng.normal(size=shape)
        cell = float(np.prod([length / n for length, n in zip(space.lengths, shape)]))
        assert space.squared_norm(x) == pytest.approx(cell * float(np.sum(x**2)))

    @pytest.mark.parametrize("shape", SHAPES)
    def test_the_lebesgue_basis_is_orthonormal(self, shape):
        space = Lebesgue(shape)
        assert space.is_orthonormal
        assert np.allclose(space.gram_matrix(), np.identity(space.dim))

    @pytest.mark.parametrize("shape", [(8,), (7,), (6, 4), (4, 4, 4)])
    def test_basis_at_reproduces_the_field(self, shape, rng):
        """``x(p) == sum_i c_i phi_i(p)``, which pins the basis functions.

        Checked at grid points, where the band-limited interpolant is the field
        itself. The Nyquist mode of an even axis is the subtle case: it is a
        fixed point of conjugation, so it appears once in the spectrum and
        carries amplitude ``1/sqrt(V)`` rather than the ``sqrt(2/V)`` of a
        conjugate pair.
        """
        space = Lebesgue(shape)
        x = rng.normal(size=shape)
        components = space.to_components(x)
        for _ in range(8):
            index = tuple(int(rng.integers(0, n)) for n in shape)
            point = np.array([axis[i] for axis, i in zip(space.grid_axes, index)])
            assert components @ space.basis_at(point) == pytest.approx(
                x[index], abs=1e-10
            )

    def test_a_point_needs_the_right_number_of_coordinates(self):
        with pytest.raises(ValueError, match="coordinates"):
            Lebesgue((4, 4)).basis_at(np.array([0.1]))


class TestSpaceAxioms:
    @pytest.mark.parametrize("shape", [(8,), (6, 4), (4, 4, 4)])
    def test_lebesgue(self, shape, rng):
        check_space(Lebesgue(shape), rng=rng, rebuild=lambda: Lebesgue(shape))
        check_coordinates(Lebesgue(shape), rng=rng)

    @pytest.mark.parametrize("shape", [(8,), (6, 4), (4, 4, 4)])
    def test_sobolev(self, shape, rng):
        space = Sobolev(shape, 2.0, 0.3)
        check_space(space, rng=rng, rebuild=lambda: Sobolev(shape, 2.0, 0.3))
        check_coordinates(space, rng=rng)

    def test_white_noise_is_white_on_a_sobolev_space(self, rng):
        check_white_noise(Sobolev((16,), 2.0, 0.3), rng=rng, samples=20000, rtol=0.08)

    def test_equality_and_hashing(self):
        assert Lebesgue((8,)) == Lebesgue((8,))
        assert Sobolev((8,), 2.0, 0.3) != Sobolev((8,), 3.0, 0.3)
        assert Sobolev((8,), 2.0, 0.3) != Lebesgue((8,))
        assert len({Lebesgue((8,)), Lebesgue((8,)), Lebesgue((4,))}) == 2


class TestSobolevMetric:
    def test_it_is_a_diagonal_metric_space(self):
        """Not a mass-weighted one, which is the simplification of DESIGN.md 13.2."""
        space = Sobolev((8,), 2.0, 0.3)
        assert space.has_diagonal_metric
        assert not space.is_orthonormal

    def test_the_metric_is_the_sobolev_symbol(self):
        space = Sobolev((8,), 2.0, 0.3)
        expected = (1.0 + 0.3**2 * space.laplacian_eigenvalues) ** 2.0
        assert np.allclose(space.metric_values, expected)

    def test_the_coordinate_map_is_shared_with_the_lebesgue_space(self, rng):
        """Same components, different metric. That is the whole difference."""
        lebesgue, sobolev = Lebesgue((8,)), Sobolev((8,), 2.0, 0.3)
        x = rng.normal(size=(8,))
        assert np.allclose(lebesgue.to_components(x), sobolev.to_components(x))
        assert sobolev.squared_norm(x) > lebesgue.squared_norm(x)

    def test_a_higher_order_penalises_roughness_more(self, rng):
        rough = np.cos(8.0 * np.pi * np.arange(32) / 32.0)
        gentle = Sobolev((32,), 1.0, 0.3)
        strict = Sobolev((32,), 3.0, 0.3)
        assert strict.norm(rough) > gentle.norm(rough)

    def test_with_order(self):
        space = Sobolev((8,), 2.0, 0.3).with_order(1.0)
        assert space.order == 1.0
        assert space.length_scale == 0.3

    def test_the_length_scale_is_not_called_scale(self):
        """``scale`` is the vector-scaling operation; see test_code_practice."""
        space = Sobolev((8,), 2.0, 0.3)
        assert space.length_scale == 0.3
        assert callable(space.scale)


class TestInvariantOperators:
    def test_the_laplacian_is_diagonal_and_semidefinite(self, rng):
        space = Sobolev((16,), 2.0, 0.3)
        laplacian = space.laplacian
        assert isinstance(laplacian, DiagonalLinearOperator)
        assert Traits.POSITIVE_SEMIDEFINITE & laplacian.traits
        check_operator(laplacian, rng=rng)
        check_traits(laplacian, rng=rng)

    def test_it_annihilates_constants(self):
        space = Lebesgue((16,))
        constant = np.ones(16)
        assert space.norm(space.laplacian(constant)) < 1e-10

    def test_it_matches_the_analytic_eigenvalue(self):
        """``-d^2/dx^2 cos(k x) == k^2 cos(k x)`` on the unit circle."""
        space = Lebesgue((32,))
        field = np.cos(3.0 * space.grid_axes[0])
        assert np.allclose(space.laplacian(field), 9.0 * field, atol=1e-10)

    def test_an_invariant_operator_stays_diagonal(self, rng):
        space = Sobolev((16,), 2.0, 0.3)
        smoother = space.invariant_operator(lambda values: np.exp(-0.1 * values))
        assert isinstance(smoother, DiagonalLinearOperator)
        assert Traits.POSITIVE_DEFINITE & smoother.traits
        check_traits(smoother, rng=rng)
        assert isinstance(smoother @ space.laplacian, DiagonalLinearOperator)

    def test_the_functional_calculus_is_exact(self, rng):
        space = Sobolev((16,), 2.0, 0.3)
        operator = space.invariant_operator(lambda values: 1.0 + values)
        root = operator.sqrt
        x = space.random(rng=rng)
        assert space.norm(space.subtract(root(root(x)), operator(x))) < 1e-10

    def test_a_symbol_of_the_wrong_length_is_refused(self):
        with pytest.raises(ValueError, match="expected"):
            Lebesgue((8,)).invariant_operator(lambda values: np.ones(3))


class TestInvariantMeasures:
    def test_the_moments_match(self, rng):
        space = Sobolev((16,), 2.0, 0.3)
        measure = space.sobolev_measure(2.0, 0.3)
        check_measure(measure, rng=rng, samples=8000, rtol=0.12)

    def test_the_covariance_is_definite_and_diagonal(self, rng):
        space = Sobolev((16,), 2.0, 0.3)
        measure = space.sobolev_measure(2.0, 0.3)
        assert Traits.POSITIVE_DEFINITE & measure.covariance.traits
        check_traits(measure.covariance, rng=rng)

    def test_a_semidefinite_measure_has_no_precision(self):
        """A measure on a subspace says so rather than carrying a stand-in."""
        space = Lebesgue((8,))
        variances = np.ones(8)
        variances[-3:] = 0.0
        measure = space.invariant_measure(variances)
        assert measure.precision is None
        assert measure.can_sample

    def test_smoother_priors_give_smoother_samples(self, rng):
        space = Lebesgue((64,))
        rough = space.sobolev_measure(1.0, 0.1).sample(rng=rng)
        smooth = space.sobolev_measure(4.0, 0.1).sample(rng=rng)

        def roughness(field):
            return space.norm(space.laplacian(field)) / max(space.norm(field), 1e-30)

        assert roughness(smooth) < roughness(rough)

    def test_a_heat_measure(self, rng):
        space = Lebesgue((16,))
        measure = space.heat_measure(0.05)
        check_measure(measure, rng=rng, samples=6000, rtol=0.15)

    def test_negative_variances_are_refused(self):
        with pytest.raises(ValueError, match="non-negative"):
            Lebesgue((8,)).invariant_measure(-np.ones(8))


class TestPointEvaluation:
    def test_it_evaluates(self, rng):
        space = Sobolev((32,), 2.0, 0.3)
        points = [np.array([0.4]), np.array([2.1]), np.array([5.0])]
        operator = space.point_evaluation_operator(points)
        field = space.project_function(lambda t: np.sin(3.0 * t))
        assert np.allclose(
            operator(field), np.sin(3.0 * np.array([0.4, 2.1, 5.0])), atol=1e-10
        )

    def test_the_adjoint_is_correct(self, rng):
        space = Sobolev((32,), 2.0, 0.3)
        operator = space.point_evaluation_operator([np.array([0.4]), np.array([2.1])])
        check_operator(operator, rng=rng)

    def test_the_adjoint_returns_dirac_representers(self, rng):
        """The operator-level form of DESIGN.md 5.6, on a real space."""
        space = Sobolev((32,), 2.0, 0.3)
        point = np.array([0.4])
        operator = space.point_evaluation_operator([point, np.array([2.1])])
        assert np.allclose(
            space.to_components(operator.adjoint(np.array([1.0, 0.0]))),
            space.to_components(space.dirac(point).representer),
        )

    def test_the_dirac_pairs_as_an_evaluation(self, rng):
        space = Sobolev((32,), 2.0, 0.3)
        point = np.array([1.3])
        functional = space.dirac(point)
        field = space.project_function(lambda t: np.cos(2.0 * t))
        assert functional(field) == pytest.approx(np.cos(2.0 * 1.3), abs=1e-10)

    def test_the_representer_differs_from_the_raw_components(self, rng):
        """Because the Sobolev metric is not the identity."""
        space = Sobolev((32,), 2.0, 0.3)
        functional = space.dirac(np.array([1.3]))
        assert not np.allclose(
            space.to_components(functional.representer), functional.matrix().ravel()
        )

    def test_no_points_is_refused(self):
        with pytest.raises(ValueError, match="At least one"):
            Lebesgue((8,)).point_evaluation_operator([])


class TestNonUniformFFT:
    """Scattered evaluation, and its adjoint, through finufft.

    Every test compares the transform route against the direct sum over the
    basis. The direct route is obviously right and unusably slow; the fast one
    is neither, so it is only ever trusted where the two agree.
    """

    SHAPES = [
        ((64,), (1.0,)),
        ((65,), (2.0,)),  # odd, so the Nyquist mode is absent
        ((32, 32), (1.0, 2.0)),
        ((8, 9), (1.0, 1.0)),  # mixed parity
        ((16, 16, 16), (1.0, 1.0, 1.0)),
    ]

    @pytest.mark.parametrize("shape,lengths", SHAPES, ids=lambda v: str(v))
    def test_evaluation_matches_the_direct_sum(self, shape, lengths, rng):
        X = Sobolev(shape, 2.0, 0.05, lengths=lengths)
        points = [
            rng.uniform(0.0, 1.0, len(shape)) * np.asarray(lengths) for _ in range(40)
        ]
        x = X.random(rng=rng)
        assert np.allclose(
            X.evaluate(x, points),
            SymmetricSpace.evaluate(X, x, points),
            atol=1e-9,
        )

    @pytest.mark.parametrize("shape,lengths", SHAPES, ids=lambda v: str(v))
    def test_accumulation_matches_the_direct_sum(self, shape, lengths, rng):
        X = Sobolev(shape, 2.0, 0.05, lengths=lengths)
        points = [
            rng.uniform(0.0, 1.0, len(shape)) * np.asarray(lengths) for _ in range(40)
        ]
        weights = rng.normal(size=40)
        assert np.allclose(
            X.accumulate(weights, points),
            SymmetricSpace.accumulate(X, weights, points),
            atol=1e-9,
        )

    def test_a_nyquist_leading_mode_survives_the_round_trip(self, rng):
        """The reason the spectrum is padded, tested on its own.

        A paired mode may sit at ``+n/2`` on a leading axis, which is outside
        the range finufft indexes. Folding it to ``-n/2`` flips the sign of its
        phase on that axis alone, which is wrong for every point off the grid
        and right for every point on it — so a grid-only test would pass.
        """
        X = Lebesgue((8, 8), lengths=(1.0, 1.0))
        wavenumbers = X._packing.wavenumbers
        offending = np.flatnonzero(wavenumbers[0] == 4)
        assert offending.size, "this shape was chosen to have such a mode"

        # Non-vacuous: at least one of these really would move if folded.
        point = np.array([0.137, 0.421])
        angles = 2.0 * np.pi * point
        folds = [
            abs(
                np.cos(wavenumbers[:, i] @ angles)
                - np.cos(-wavenumbers[0, i] * angles[0] + wavenumbers[1, i] * angles[1])
            )
            for i in offending
        ]
        assert max(folds) > 0.5

        for index in offending:
            components = np.zeros(X.dim)
            components[index] = 1.0
            field = X.from_components(components)
            points = [rng.uniform(0.0, 1.0, 2) for _ in range(12)]
            direct = np.array([float(X.basis_at(point)[index]) for point in points])
            assert np.allclose(X.evaluate(field, points), direct, atol=1e-9)

    def test_the_operator_agrees_with_the_assembled_one(self, rng):
        X = Sobolev((32, 32), 2.0, 0.05, lengths=(1.0, 1.0))
        points = [rng.uniform(0.0, 1.0, 2) for _ in range(6)]
        A = X.point_evaluation_operator(points)
        check_operator(A, rng=rng)
        x = X.random(rng=rng)
        assert np.allclose(A(x), A.assembled()(x), atol=1e-9)

    def test_four_dimensions_fall_back(self, rng):
        """finufft stops at three; the answer must not."""
        X = Lebesgue((6, 6, 6, 6))
        assert X._nufft_layout is None
        points = [rng.uniform(0.0, 1.0, 4) for _ in range(5)]
        x = X.random(rng=rng)
        assert np.allclose(X.evaluate(x, points), SymmetricSpace.evaluate(X, x, points))
        check_operator(X.point_evaluation_operator(points), rng=rng)

    def test_a_wrong_point_shape_is_refused(self):
        X = Lebesgue((16, 16))
        with pytest.raises(ValueError, match="coordinates each"):
            X.evaluate(X.zero(), [np.array([0.1, 0.2, 0.3])])


class TestFormalAdjointLift:
    def test_the_action_is_unchanged(self, rng):
        """Define on L2, use on the Sobolev space. DESIGN.md 3.5 and 13.2."""
        lebesgue, sobolev = Lebesgue((16,)), Sobolev((16,), 2.0, 0.3)
        matrix = rng.normal(size=(16, 16))
        base = LinearOperator.from_matrix(lebesgue, lebesgue, matrix, form="components")
        lifted = lift_formal_adjoint(base, sobolev)

        x = sobolev.random(rng=rng)
        assert np.allclose(
            sobolev.to_components(lifted(x)),
            matrix @ sobolev.to_components(x),
        )

    def test_the_adjoint_is_correct_under_the_new_metric(self, rng):
        lebesgue, sobolev = Lebesgue((16,)), Sobolev((16,), 2.0, 0.3)
        matrix = rng.normal(size=(16, 16))
        base = LinearOperator.from_matrix(lebesgue, lebesgue, matrix, form="components")
        check_operator(lift_formal_adjoint(base, sobolev), rng=rng)

    def test_it_claims_no_symmetry(self, rng):
        """A formally self-adjoint operator need not stay self-adjoint.

        It does only if it commutes with the ratio of the two metrics, which
        for a general operator it does not. See DESIGN.md 3.5.
        """
        lebesgue, sobolev = Lebesgue((16,)), Sobolev((16,), 2.0, 0.3)
        matrix = rng.normal(size=(16, 16))
        matrix = matrix + matrix.T
        base = LinearOperator.from_matrix(
            lebesgue, lebesgue, matrix, traits=Traits.SELF_ADJOINT, form="components"
        )
        lifted = lift_formal_adjoint(base, sobolev)
        assert lifted.traits == Traits.NONE

        x, y = sobolev.random(rng=rng), sobolev.random(rng=rng)
        assert not np.isclose(
            sobolev.inner_product(lifted(x), y),
            sobolev.inner_product(x, lifted(y)),
        )

    def test_a_dimension_mismatch_is_refused(self, rng):
        lebesgue, sobolev = Lebesgue((8,)), Sobolev((16,), 2.0, 0.3)
        base = LinearOperator.from_matrix(
            lebesgue, lebesgue, np.identity(8), form="components"
        )
        with pytest.raises(ValueError, match="dimension"):
            lift_formal_adjoint(base, sobolev)


class TestConstruction:
    def test_defaults_give_the_unit_circle(self):
        space = Lebesgue((8,))
        assert space.spatial_dimension == 1
        assert space.volume == pytest.approx(2.0 * np.pi)

    def test_lengths_are_respected(self):
        space = Lebesgue((8, 4), lengths=(2.0, 3.0))
        assert space.volume == pytest.approx(6.0)
        assert space.spatial_dimension == 2

    def test_a_three_dimensional_box(self):
        """The case v1 does not have at all."""
        space = Sobolev((8, 8, 8), 2.0, 0.2, lengths=(1.0, 1.0, 1.0))
        assert space.spatial_dimension == 3
        assert space.dim == 512

    @pytest.mark.parametrize(
        "shape, kwargs, message",
        [
            ((1,), {}, "at least two"),
            ((), {}, "at least two"),
            ((4,), dict(lengths=(1.0, 2.0)), "lengths for"),
            ((4,), dict(lengths=(-1.0,)), "must be positive"),
            ((4,), dict(length_scale=0.0), "length_scale"),
        ],
    )
    def test_bad_arguments_are_refused(self, shape, kwargs, message):
        with pytest.raises(ValueError, match=message):
            PeriodicBox(shape, **kwargs)

    def test_pointwise_multiplication(self, rng):
        space = Lebesgue((16,))
        x, y = rng.normal(size=16), rng.normal(size=16)
        assert np.allclose(space.pointwise_multiply(x, y), x * y)
