"""Tensor products of coordinate spaces: the space, the contraction with a
functional on one factor, the tensor product of operators, and the Gaussian
whose covariance is a tensor product; on factors whose metrics are dense as
well as diagonal, since a Kronecker Gram applied along the wrong axis is
invisible on a diagonal one."""

import numpy as np
import pytest

from pygeoinf.algebra.diagonal import DiagonalLinearOperator
from pygeoinf.algebra.operators import LinearFunctional, LinearOperator
from pygeoinf.algebra.spaces import EuclideanSpace, MassWeightedSpace
from pygeoinf.algebra.tensor_product import (
    TensorProduct,
    TensorProductLinearOperator,
)
from pygeoinf.probability import GaussianMeasure
from pygeoinf.testing import (
    check_coordinates,
    check_measure,
    check_operator,
    check_space,
    check_traits,
    check_white_noise,
)
from pygeoinf.traits import Traits

from .conftest import (
    make_dense_metric_space,
    make_weighted_space,
    nonsymmetric,
    spd_matrix,
    spd_operator,
)
from .doubles import OpaqueSpace

BUILDERS = {
    "weighted-dense": lambda: [make_weighted_space(), make_dense_metric_space(3)],
    "three": lambda: [
        make_dense_metric_space(3),
        make_weighted_space(),
        EuclideanSpace(2),
    ],
}


@pytest.fixture(params=sorted(BUILDERS), ids=sorted(BUILDERS))
def build(request):
    """A builder of the factors, two with a dense metric among them and
    three with a Euclidean one, so that an axis handled wrongly shows."""
    return BUILDERS[request.param]


@pytest.fixture
def P(build):
    return TensorProduct(build())


def kron(matrices):
    result = matrices[0]
    for matrix in matrices[1:]:
        result = np.kron(result, matrix)
    return result


class TestTensorProductSpace:
    def test_axioms(self, P, build, rng):
        check_space(P, rng=rng, rebuild=lambda: TensorProduct(build()))
        check_coordinates(P, rng=rng)
        assert P.shape == tuple(factor.dim for factor in P.factors)
        assert P.dim == int(np.prod(P.shape))
        assert P.zero().shape == P.shape

    def test_white_noise_has_identity_covariance(self, P, rng):
        """The whitening along each axis composes to the product's."""
        check_white_noise(P, rng=rng)

    def test_the_gram_matrix_is_the_kronecker_product(self, P):
        expected = kron([factor.gram_matrix() for factor in P.factors])
        np.testing.assert_allclose(P.gram_matrix(), expected, rtol=1e-12)
        np.testing.assert_allclose(P.gram_diagonal(), np.diag(expected), rtol=1e-12)
        columns = np.identity(P.dim)
        np.testing.assert_allclose(
            P.apply_gram_to_columns(columns), expected, rtol=1e-12
        )
        np.testing.assert_allclose(
            P.solve_gram_to_columns(expected), columns, rtol=1e-10, atol=1e-12
        )

    def test_the_metric_is_the_product_on_rank_one_vectors(self, P, rng):
        """``(x_1 (x) x_2, y_1 (x) y_2) == (x_1, y_1) (x_2, y_2)``."""
        xs = [factor.random(rng=rng) for factor in P.factors]
        ys = [factor.random(rng=rng) for factor in P.factors]
        expected = np.prod(
            [f.inner_product(x, y) for f, x, y in zip(P.factors, xs, ys)]
        )
        np.testing.assert_allclose(
            P.inner_product(P.rank_one(*xs), P.rank_one(*ys)), expected, rtol=1e-12
        )
        with pytest.raises(ValueError):
            P.rank_one(*xs[:-1])

    def test_the_metric_is_diagonal_exactly_when_every_factor_is(self):
        diagonal = TensorProduct([make_weighted_space(), EuclideanSpace(2)])
        assert diagonal.has_diagonal_metric
        assert not diagonal.is_orthonormal
        np.testing.assert_allclose(
            diagonal.metric_values, np.kron(np.array([1.0, 4.0, 9.0, 0.25]), [1, 1])
        )
        assert TensorProduct([EuclideanSpace(2), EuclideanSpace(3)]).is_orthonormal
        mixed = TensorProduct([make_weighted_space(), make_dense_metric_space(3)])
        assert not mixed.has_diagonal_metric
        with pytest.raises(ValueError):
            mixed.metric_values

    def test_equality_is_by_the_factors_in_order(self):
        A, B = make_weighted_space(), make_dense_metric_space(3)
        assert TensorProduct([A, B]) == TensorProduct([A, B])
        assert hash(TensorProduct([A, B])) == hash(TensorProduct([A, B]))
        assert TensorProduct([A, B]) != TensorProduct([B, A])
        assert not TensorProduct([A, B]).is_function_space

    def test_refuses_one_factor_and_a_factor_without_coordinates(self):
        with pytest.raises(ValueError):
            TensorProduct([make_weighted_space()])
        with pytest.raises(TypeError):
            TensorProduct([make_weighted_space(), OpaqueSpace(np.ones(3))])

    def test_a_coupled_norm_is_a_mass_weighting_of_the_product(self, rng):
        """A norm that does not factor, an isotropic one on the product
        basis say, is another inner product on the same vectors: a mass
        operator on the product, under which every coordinate routine still
        runs."""
        P = TensorProduct([make_weighted_space(), EuclideanSpace(3)])
        coupling = 1.0 + np.add.outer(np.arange(4.0), np.arange(3.0)).reshape(-1)
        weighted = MassWeightedSpace(P, DiagonalLinearOperator(P, coupling))
        check_space(weighted, rng=rng)
        check_coordinates(weighted, rng=rng)
        assert weighted.has_diagonal_metric
        np.testing.assert_allclose(
            weighted.gram_diagonal(), P.metric_values * coupling, rtol=1e-12
        )


class TestContraction:
    def test_is_the_functional_applied_on_one_factor(self, P, rng):
        """On a rank-one vector the contraction with ``f`` on factor ``i`` is
        ``f(x_i)`` times the product of the other factors' vectors, and the
        codomain is the remaining factor itself when one is left."""
        xs = [factor.random(rng=rng) for factor in P.factors]
        for axis, factor in enumerate(P.factors):
            functional = LinearFunctional.from_representer(
                factor, factor.random(rng=rng)
            )
            operator = P.contraction(functional)
            check_operator(operator, rng=rng)
            rest = [x for i, x in enumerate(xs) if i != axis]
            if len(rest) == 1:
                assert operator.codomain == P.factors[1 - axis]
                expected = operator.codomain.scale(functional(xs[axis]), rest[0])
                np.testing.assert_allclose(
                    operator.codomain.to_components(operator(P.rank_one(*xs))),
                    operator.codomain.to_components(expected),
                    rtol=1e-12,
                )
            else:
                assert isinstance(operator.codomain, TensorProduct)
                expected = functional(xs[axis]) * operator.codomain.rank_one(*rest)
                np.testing.assert_allclose(
                    operator(P.rank_one(*xs)), expected, rtol=1e-12
                )

    def test_the_adjoint_is_the_representer_tensored_with_the_data(self, rng):
        A, B = make_dense_metric_space(3), make_weighted_space()
        P = TensorProduct([A, B])
        functional = LinearFunctional.from_representer(A, A.random(rng=rng))
        y = B.random(rng=rng)
        np.testing.assert_allclose(
            P.contraction(functional).adjoint(y),
            P.rank_one(functional.representer, y),
            rtol=1e-12,
        )

    def test_the_axis_is_found_or_asked_for(self, rng):
        A, B = make_weighted_space(), make_dense_metric_space(3)
        on_A = LinearFunctional.from_representer(A, A.random(rng=rng))
        on_B = LinearFunctional.from_representer(B, B.random(rng=rng))
        repeated = TensorProduct([A, B, A])
        with pytest.raises(ValueError, match="axis="):
            repeated.contraction(on_A)
        check_operator(repeated.contraction(on_A, axis=2), rng=rng)
        with pytest.raises(ValueError):
            repeated.contraction(on_A, axis=1)
        with pytest.raises(ValueError):
            repeated.contraction(on_A, axis=3)
        with pytest.raises(ValueError):
            TensorProduct([A, A]).contraction(on_B)


class TestTensorProductLinearOperator:
    def test_is_the_kronecker_product_of_the_factors(self, P, rng):
        operators = []
        matrices = []
        for factor in P.factors:
            if isinstance(factor, EuclideanSpace):
                operators.append(LinearOperator.identity(factor))
                matrices.append(np.identity(factor.dim))
            else:
                operator, matrix = nonsymmetric(factor, rng)
                operators.append(operator)
                matrices.append(matrix)
        K = TensorProductLinearOperator(operators)
        assert K.domain == P and K.codomain == P
        check_operator(K, rng=rng)
        np.testing.assert_allclose(K.matrix(form="components"), kron(matrices))
        np.testing.assert_allclose(
            K.adjoint.matrix(form="galerkin"), K.matrix(form="galerkin").T
        )
        assert isinstance(K.adjoint, TensorProductLinearOperator)
        assert K.adjoint.adjoint is K
        x = P.random(rng=rng)
        np.testing.assert_allclose(
            P.to_components(K(x)), kron(matrices) @ P.to_components(x)
        )

    def test_the_traits_are_those_every_factor_claims(self, rng):
        A, B = make_dense_metric_space(3), make_weighted_space()
        definite = TensorProductLinearOperator(
            [spd_operator(A, rng), spd_operator(B, rng)]
        )
        assert Traits.POSITIVE_DEFINITE & definite.traits
        check_traits(definite, rng=rng)
        mixed = TensorProductLinearOperator(
            [nonsymmetric(A, rng)[0], spd_operator(B, rng)]
        )
        assert mixed.traits == Traits.NONE
        check_operator(mixed, rng=rng)
        with pytest.raises(ValueError):
            TensorProductLinearOperator([spd_operator(A, rng)])

    def test_the_identity_factors_cost_nothing_and_a_lift_is_one_per_column(self, rng):
        """``I (x) B`` applies ``B`` once per basis vector of the identity's
        factor, and the lift of ``B`` is that operator."""
        A, B = make_weighted_space(), make_dense_metric_space(3)
        P = TensorProduct([A, B])
        count = 0
        base = spd_operator(B, rng)

        def counted(x):
            nonlocal count
            count += 1
            return base(x)

        operator = LinearOperator.from_callables(B, B, counted, adjoint=counted)
        lifted = P.lift(operator)
        assert isinstance(lifted, TensorProductLinearOperator)
        assert lifted.operators[0].is_identity
        x = P.random(rng=rng)
        image = lifted(x)
        assert count == A.dim
        np.testing.assert_allclose(image, P.lift(base)(x), rtol=1e-12)
        a, b = A.random(rng=rng), B.random(rng=rng)
        np.testing.assert_allclose(
            lifted(P.rank_one(a, b)), P.rank_one(a, base(b)), rtol=1e-12
        )
        with pytest.raises(ValueError):
            TensorProduct([B, B]).lift(base)

    def test_composition_scaling_and_blocks_stay_structured(self, rng):
        A, B = make_dense_metric_space(3), make_weighted_space()
        first = TensorProductLinearOperator(
            [nonsymmetric(A, rng)[0], spd_operator(B, rng)]
        )
        second = TensorProductLinearOperator(
            [spd_operator(A, rng), nonsymmetric(B, rng)[0]]
        )
        composed = first @ second
        assert isinstance(composed, TensorProductLinearOperator)
        check_operator(composed, rng=rng)
        x = first.domain.random(rng=rng)
        np.testing.assert_allclose(composed(x), first(second(x)), rtol=1e-12)
        scaled = 3.0 * first
        assert isinstance(scaled, TensorProductLinearOperator)
        np.testing.assert_allclose(scaled(x), 3.0 * first(x), rtol=1e-12)
        gram = first @ first.adjoint
        assert isinstance(gram, TensorProductLinearOperator)
        assert Traits.POSITIVE_SEMIDEFINITE & gram.traits
        check_traits(gram, rng=rng)

    def test_diagonal_factors_give_a_scaling_the_measures_can_read(self, rng):
        A, B = make_weighted_space(), EuclideanSpace(2)
        K = TensorProductLinearOperator(
            [
                DiagonalLinearOperator(A, np.array([1.0, 2.0, 3.0, 4.0])),
                DiagonalLinearOperator(B, np.array([5.0, 6.0])),
            ]
        )
        assert Traits.POSITIVE_DEFINITE & K.traits
        check_traits(K, rng=rng)
        np.testing.assert_allclose(
            K._known_scaling(), np.kron([1.0, 2.0, 3.0, 4.0], [5.0, 6.0])
        )
        np.testing.assert_allclose(
            np.diag(K.matrix(form="components")), K._known_scaling()
        )


class TestTensorProductGaussian:
    def test_the_covariance_is_the_tensor_product_of_the_factors(self, rng):
        A, B = make_dense_metric_space(3), make_weighted_space()
        first = GaussianMeasure.from_covariance_matrix(
            A, spd_matrix(rng, 3), form="galerkin"
        )
        second = GaussianMeasure.from_covariance_matrix(
            B, spd_matrix(rng, 4), form="galerkin"
        )
        measure = GaussianMeasure.from_tensor_product([first, second])
        assert measure.domain == TensorProduct([A, B])
        assert isinstance(measure.covariance, TensorProductLinearOperator)
        assert measure.can_sample
        assert measure.precision is not None
        assert measure.precision_factor is not None
        np.testing.assert_allclose(
            measure.covariance.matrix(form="components"),
            np.kron(
                first.covariance.matrix(form="components"),
                second.covariance.matrix(form="components"),
            ),
        )
        check_measure(measure, rng=rng, samples=4000, rtol=0.2)
        assert measure.sample(rng=rng).shape == (3, 4)

    def test_diagonal_factors_on_a_diagonal_product_give_a_diagonal_covariance(
        self, rng
    ):
        A, B = EuclideanSpace(2), make_weighted_space()
        first = GaussianMeasure.from_standard_deviation(A, 2.0)
        second = GaussianMeasure(
            B, covariance=DiagonalLinearOperator(B, np.array([1.0, 2.0, 3.0, 4.0]))
        )
        measure = GaussianMeasure.from_tensor_product([first, second])
        assert isinstance(measure.covariance, DiagonalLinearOperator)
        np.testing.assert_allclose(
            measure.covariance.eigenvalues, np.kron([4.0, 4.0], [1.0, 2.0, 3.0, 4.0])
        )
        assert measure.can_sample and measure.precision is not None
        check_measure(measure, rng=rng, samples=4000, rtol=0.2)

    def test_the_mean_is_the_products_and_the_factors_are_centered(self, rng):
        A, B = make_weighted_space(), EuclideanSpace(2)
        P = TensorProduct([A, B])
        first = GaussianMeasure(
            A, covariance=DiagonalLinearOperator(A, np.array([1.0, 2.0, 3.0, 4.0]))
        )
        second = GaussianMeasure.from_standard_deviation(B, 1.0)
        mean = P.random(rng=rng)
        measure = GaussianMeasure.from_tensor_product(
            [first, second], domain=P, expectation=mean
        )
        np.testing.assert_allclose(measure.expectation, mean)
        shifted = GaussianMeasure(
            A,
            expectation=A.random(rng=rng),
            covariance=DiagonalLinearOperator(A, np.ones(4)),
        )
        with pytest.raises(ValueError, match="centered"):
            GaussianMeasure.from_tensor_product([shifted, second])
        with pytest.raises(ValueError):
            GaussianMeasure.from_tensor_product([first])
        with pytest.raises(ValueError):
            GaussianMeasure.from_tensor_product(
                [first, second], domain=TensorProduct([B, A])
            )


class TestFieldsThroughTime:
    """The case the product exists for: a Sobolev space on an interval of
    time and a Sobolev space on the sphere."""

    @pytest.fixture(scope="class")
    def spaces(self):
        randomfield = pytest.importorskip("planetmodel.randomfield")
        if not hasattr(randomfield, "SpectralBasis"):  # pragma: no cover
            pytest.skip("needs planetmodel 1.2 or later")
        from pygeoinf.interval import Lebesgue as TimeLebesgue
        from pygeoinf.interval import Sobolev as TimeSobolev
        from pygeoinf.symmetric_space.sphere import Lebesgue as SphereLebesgue
        from pygeoinf.symmetric_space.sphere import Sobolev as SphereSobolev

        return {
            "time": lambda modes=16: TimeSobolev(
                modes, 1.0, 2.0, lower=0.0, upper=26.0
            ),
            "time_lebesgue": lambda modes=16: TimeLebesgue(
                modes, lower=0.0, upper=26.0, length_scale=2.0
            ),
            "field": lambda: SphereSobolev(8, 2.0, 0.3),
            "field_lebesgue": lambda: SphereLebesgue(4),
        }

    def test_the_history_space(self, spaces, rng):
        H = TensorProduct([spaces["time"](), spaces["field"]()])
        assert H.has_diagonal_metric and H.is_function_space
        check_space(
            H,
            rng=rng,
            rebuild=lambda: TensorProduct([spaces["time"](), spaces["field"]()]),
        )
        check_coordinates(H, rng=rng)

    def test_evaluation_at_a_time_is_the_contraction_with_the_dirac(self, spaces, rng):
        """The field at a time is the contraction with the time space's
        Dirac functional, which refuses a time space of order one half or
        below; the adjoint carries the time metric and not the field's."""
        T, S = spaces["time"](), spaces["field"]()
        H = TensorProduct([T, S])
        evaluation = H.contraction(T.dirac(9.3))
        assert evaluation.codomain == S
        check_operator(evaluation, rng=rng)
        a, f = T.random(rng=rng), S.random(rng=rng)
        field = evaluation(H.rank_one(a, f))
        np.testing.assert_allclose(
            field.data, T.evaluate(a, [9.3])[0] * f.data, rtol=1e-10, atol=1e-13
        )
        with pytest.raises(ValueError):
            spaces["time_lebesgue"]().dirac(9.3)

    def test_evaluation_is_bounded_above_order_one_half(self, spaces, rng):
        """With the time order one, the representer of evaluation at a time
        converges as time modes are added, to the Green's function of
        ``1 - L^2 d^2/dt^2`` at zero lag, ``1 / (2 L)``; in ``L^2`` it grows
        with the modes."""
        F = spaces["field_lebesgue"]()
        field = F.random(rng=rng)

        def ratio(time_space):
            H = TensorProduct([time_space, F])
            representer = H.contraction(time_space.dirac(13.0, unsafe=True)).adjoint(
                field
            )
            return H.squared_norm(representer) / F.squared_norm(field)

        sobolev = [ratio(spaces["time"](modes)) for modes in (32, 64, 128)]
        lebesgue = [ratio(spaces["time_lebesgue"](modes)) for modes in (32, 64, 128)]
        changes = np.diff(sobolev)
        assert changes[1] < 0.6 * changes[0]
        assert abs(sobolev[-1] - 1.0 / 4.0) < 0.05 / 4.0
        assert lebesgue[-1] > 3.5 * lebesgue[0]

    def test_a_separable_sobolev_prior_is_diagonal_and_samples(self, spaces, rng):
        T, S = spaces["time"](), spaces["field"]()
        H = TensorProduct([T, S])
        prior = GaussianMeasure.from_tensor_product(
            [T.sobolev_measure(1.0), S.sobolev_measure(2.0, 0.3)], domain=H
        )
        assert isinstance(prior.covariance, DiagonalLinearOperator)
        assert prior.precision is not None
        draw = prior.sample(rng=rng)
        assert draw.shape == H.shape
        field = H.contraction(T.dirac(9.3))(draw)
        assert field.lmax == S.lmax

    def test_an_operator_at_each_time_is_lifted(self, spaces, rng):
        T, S = spaces["time"](), spaces["field"]()
        H = TensorProduct([T, S])
        B = S.sobolev_measure(2.0, 0.3).covariance
        lifted = H.lift(B)
        assert Traits.POSITIVE_DEFINITE & lifted.traits
        check_operator(lifted, rng=rng)
        a, f = T.random(rng=rng), S.random(rng=rng)
        np.testing.assert_allclose(
            lifted(H.rank_one(a, f)), H.rank_one(a, B(f)), rtol=1e-10
        )
