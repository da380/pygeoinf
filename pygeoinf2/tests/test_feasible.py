"""
The feasible property set in the geometry layer: the closed form, its
generalisation to ellipsoid priors and centres, the dispatch by form, and
the conversions between a set's descriptions.
"""

import numpy as np
import pytest

from pygeoinf2.algebra.operators import LinearOperator
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.geometry import (
    Ball,
    ClosedForm,
    ConvexSet,
    Ellipsoid,
    FeasiblePropertySet,
    InclusionResult,
    SupportResult,
)
from pygeoinf2.inference import BackusGilbertParker, LinearForwardProblem
from pygeoinf2.traits import Traits

from .conftest import make_dense_metric_space, make_weighted_space


def directions(space):
    return [
        space.scale(sign, space.basis_vector(index))
        for index in range(space.dim)
        for sign in (1.0, -1.0)
    ]


def spd_operator(space, rng, *, scale=1.0):
    """A random positive definite operator and its inverse, metric and all."""
    root = rng.normal(size=(space.dim, space.dim))
    galerkin = scale * (root @ root.T + space.dim * np.identity(space.dim))
    traits = Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE
    precision = LinearOperator.from_matrix(
        space, space, galerkin, form="galerkin", traits=traits
    )
    components = np.linalg.solve(space.gram_matrix(), galerkin)
    covariance = LinearOperator.from_matrix(
        space, space, np.linalg.inv(components), form="components", traits=traits
    )
    return precision, covariance


@pytest.fixture
def setting(rng):
    """A weighted model space, an under-determined map, a property, a feasible truth."""
    model = make_weighted_space()
    data = EuclideanSpace(2)
    property_space = EuclideanSpace(2)
    forward = LinearOperator.from_matrix(
        model, data, rng.normal(size=(data.dim, model.dim)), form="galerkin"
    )
    target = LinearOperator.from_matrix(
        model,
        property_space,
        rng.normal(size=(property_space.dim, model.dim)),
        form="galerkin",
    )
    raw = model.random(rng=rng)
    truth = model.scale(2.0 / model.norm(raw), raw)
    return model, forward, target, truth, forward(truth)


class TestTheClosedFormAgreesWithThePresentRoute:
    """On a ball prior and exact data the new set must reproduce the old."""

    def test_support_membership_and_the_centre(self, setting):
        model, forward, target, truth, data = setting
        prior = Ball(model, radius=3.0)
        new = FeasiblePropertySet(forward, target, prior, data)
        old = BackusGilbertParker(LinearForwardProblem(forward), target, prior)(data)
        assert new.form == "closed_form"
        assert new.algorithms == {
            "support": ("closed_form",),
            "inclusion": ("closed_form",),
            "feasibility": ("closed_form",),
        }
        for q in directions(target.codomain):
            assert new.support(q) == pytest.approx(old.support(q), rel=1e-10)
            inner, outer = new.support_bracket(q)
            assert inner == outer
        assert np.allclose(new.ellipsoid.center, old.ellipsoid.center)
        assert new.contains(target(truth)) and old.contains(target(truth))
        assert not new.is_empty()
        assert new.inclusion_level(target(truth)) == pytest.approx(
            old.inclusion_level(target(truth)), rel=1e-8
        )
        assert model.norm(
            model.subtract(new.fitting_model(), old.fitting_model())
        ) <= 1e-10 * model.norm(truth)

    def test_the_results_carry_their_by_products(self, setting):
        model, forward, target, truth, data = setting
        feasible = FeasiblePropertySet(forward, target, Ball(model, radius=3.0), data)
        q = target.codomain.basis_vector(0)
        result = feasible.support_result(q)
        assert isinstance(result, SupportResult)
        assert result.algorithm == "closed_form" and result.converged
        extremal = result.extremal_model
        # The extremal model fits the data exactly, saturates the prior, and
        # attains the support.
        assert forward.codomain.norm(
            forward.codomain.subtract(forward(extremal), data)
        ) <= 1e-8 * forward.codomain.norm(data)
        assert model.norm(extremal) == pytest.approx(3.0, rel=1e-8)
        assert target.codomain.inner_product(q, target(extremal)) == pytest.approx(
            result.value, rel=1e-10
        )
        assert feasible.support_maximizer(q) is not None
        inclusion = feasible.inclusion_result(target(truth))
        assert isinstance(inclusion, InclusionResult)
        fitting = inclusion.fitting_model
        assert np.allclose(target(fitting), target(truth), atol=1e-8)
        assert model.norm(fitting) ** 2 == pytest.approx(inclusion.level, rel=1e-8)

    def test_data_the_prior_cannot_fit_give_an_empty_set(self, setting):
        model, forward, target, truth, data = setting
        feasible = FeasiblePropertySet(forward, target, Ball(model, radius=0.1), data)
        assert feasible.is_empty()
        with pytest.raises(ValueError, match="empty"):
            feasible.fitting_model()
        with pytest.raises(ValueError):
            feasible.support(target.codomain.basis_vector(0))


class TestTheClosedFormOnEllipsoidsAndCentres:
    """The generalisation: a quadratic prior in its own inner product, anywhere."""

    def test_an_ellipsoid_prior_is_a_ball_after_whitening(self, rng):
        """On a Euclidean space the set for (A, T, {(Q u, u) <= 1}) is the set
        for (A L^-T, T L^-T, unit ball) with Q == L L^T."""
        model, data_space, prop = (
            EuclideanSpace(12),
            EuclideanSpace(4),
            EuclideanSpace(3),
        )
        A = rng.normal(size=(4, 12))
        T = rng.normal(size=(3, 12))
        root = rng.normal(size=(12, 12))
        Q = root @ root.T + 12.0 * np.identity(12)
        L = np.linalg.cholesky(Q)
        z = rng.normal(size=12)
        inside = np.linalg.solve(L.T, 0.5 * z / np.linalg.norm(z))  # (Q u, u) == 0.25
        data = A @ inside
        pd = Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE
        ellipsoid = Ellipsoid(
            model,
            LinearOperator.from_matrix(model, model, Q, form="galerkin", traits=pd),
            covariance=LinearOperator.from_matrix(
                model, model, np.linalg.inv(Q), form="galerkin", traits=pd
            ),
        )
        general = FeasiblePropertySet(
            LinearOperator.from_matrix(model, data_space, A, form="galerkin"),
            LinearOperator.from_matrix(model, prop, T, form="galerkin"),
            ellipsoid,
            data,
        )
        whitened = FeasiblePropertySet(
            LinearOperator.from_matrix(
                model, data_space, np.linalg.solve(L, A.T).T, form="galerkin"
            ),
            LinearOperator.from_matrix(
                model, prop, np.linalg.solve(L, T.T).T, form="galerkin"
            ),
            Ball(model, radius=1.0),
            data,
        )
        assert general.form == "closed_form"
        for q in directions(prop):
            assert general.support(q) == pytest.approx(whitened.support(q), rel=1e-9)
        p = prop.random(rng=rng)
        assert general.inclusion_level(p) == pytest.approx(
            whitened.inclusion_level(p), rel=1e-8
        )
        assert general.level == 1.0 and whitened.level == 1.0

    def test_a_prior_centre_shifts_the_answer(self, rng):
        """The set with centre a and data d is the set with centre 0 and data
        d − A a, translated by T a."""
        model = make_dense_metric_space(8)
        data_space, prop = EuclideanSpace(3), EuclideanSpace(2)
        forward = LinearOperator.from_matrix(
            model, data_space, rng.normal(size=(3, 8)), form="galerkin"
        )
        target = LinearOperator.from_matrix(
            model, prop, rng.normal(size=(2, 8)), form="galerkin"
        )
        center = model.random(rng=rng)
        data = data_space.random(rng=rng)
        shifted = data_space.subtract(data, forward(center))
        centred = FeasiblePropertySet(
            forward, target, Ball(model, radius=2.0, center=center), data
        )
        origin = FeasiblePropertySet(forward, target, Ball(model, radius=2.0), shifted)
        offset = target(center)
        for q in directions(prop):
            assert centred.support(q) == pytest.approx(
                origin.support(q) + prop.inner_product(q, offset), rel=1e-9
            )
        p = prop.random(rng=rng)
        assert centred.contains(prop.add(p, offset)) == origin.contains(p)
        # The old route ignored the centre; this one does not.
        assert not centred.is_empty()


class TestDispatch:
    def test_via_by_name_instance_and_refusal(self, setting):
        model, forward, target, truth, data = setting
        feasible = FeasiblePropertySet(forward, target, Ball(model, radius=3.0), data)
        q = target.codomain.basis_vector(1)
        assert feasible.support(q, via="closed_form") == feasible.support(q)
        assert feasible.support(q, via=ClosedForm()) == feasible.support(q)
        with pytest.raises(ValueError, match="Unknown algorithm"):
            feasible.support(q, via="bisection")
        with pytest.raises(TypeError):
            feasible.support(q, via=3)

    def test_a_form_no_algorithm_answers_is_refused_by_name(self, setting):
        model, forward, target, truth, data = setting
        noisy = FeasiblePropertySet(
            forward,
            target,
            Ball(model, radius=3.0),
            data,
            noise=Ball(forward.codomain, radius=0.1),
        )
        assert noisy.form == "quadratic"
        assert noisy.algorithms["support"] == ()
        with pytest.raises(NotImplementedError, match="quadratic"):
            noisy.support(target.codomain.basis_vector(0))
        with pytest.raises(ValueError, match="does not apply"):
            noisy.support(target.codomain.basis_vector(0), via="closed_form")
        assert not noisy.has_support_function and not noisy.has_membership

    def test_a_zero_radius_noise_ball_is_exact_data(self, setting):
        model, forward, target, truth, data = setting
        feasible = FeasiblePropertySet(
            forward,
            target,
            Ball(model, radius=3.0),
            data,
            noise=Ball(forward.codomain, radius=0.0),
        )
        assert feasible.form == "closed_form" and feasible.noise is None


class TestSharingAndCaching:
    def test_the_shape_is_built_once_for_every_data_set(self, setting):
        model, forward, target, truth, data = setting
        builds = []
        original = ClosedForm._build

        def counting(self, structure):
            builds.append(structure)
            return original(self, structure)

        ClosedForm._build = counting
        try:
            feasible = FeasiblePropertySet(
                forward, target, Ball(model, radius=3.0), data
            )
            sibling = feasible.with_data(forward.codomain.scale(0.5, data))
            for q in directions(target.codomain):
                feasible.support(q)
                sibling.support(q)
            feasible.contains(target(truth))
            sibling.is_empty()
        finally:
            ClosedForm._build = original
        assert len(builds) == 1
        assert feasible._structure is sibling._structure

    def test_a_probe_is_computed_once_per_direction(self, setting):
        model, forward, target, truth, data = setting
        feasible = FeasiblePropertySet(forward, target, Ball(model, radius=3.0), data)
        q = target.codomain.basis_vector(0)
        first = feasible.support_result(q)
        again = feasible.support_result(
            target.codomain.copy(q)
        )  # same value, new object
        assert again is first


class TestTheProbes:
    def test_the_polytope_contains_the_truth_and_the_inner_hull_lies_inside(
        self, setting
    ):
        model, forward, target, truth, data = setting
        feasible = FeasiblePropertySet(forward, target, Ball(model, radius=3.0), data)
        prop = target.codomain
        angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
        qs = [prop.from_components(np.array([np.cos(a), np.sin(a)])) for a in angles]
        outer = feasible.polytope(qs)
        inner = feasible.inner_hull(qs, source="contact")
        assert outer.is_outer and not inner.is_outer
        assert outer.contains(target(truth))
        for q in qs:
            contact = feasible.support_maximizer(q)
            assert outer.contains(contact, rtol=1e-8)
            assert feasible.contains(contact, rtol=1e-8)
        assert not feasible.outside(target(truth), qs)
        far = prop.from_components(np.array([50.0, 50.0]))
        assert feasible.outside(far, qs)

    def test_the_extent_meets_the_support_on_a_scalar_property(self, setting):
        model, forward, target, truth, data = setting
        scalar = EuclideanSpace(1)
        row = LinearOperator.from_matrix(
            target.codomain, scalar, np.array([[1.0, 0.5]]), form="galerkin"
        )
        feasible = FeasiblePropertySet(
            forward, row @ target, Ball(model, radius=3.0), data
        )
        e = scalar.basis_vector(0)
        lower, upper = feasible.extent(e)
        assert upper == pytest.approx(feasible.support(e), rel=1e-6)
        assert lower == pytest.approx(
            -feasible.support(scalar.scale(-1.0, e)), rel=1e-6
        )

    def test_the_level_function_has_the_right_gradient(self, setting, rng):
        model, forward, target, truth, data = setting
        feasible = FeasiblePropertySet(forward, target, Ball(model, radius=3.0), data)
        f = feasible.level_function()
        p = target.codomain.random(rng=rng)
        gradient = f.gradient(p)
        h = 1e-6
        for i in range(2):
            e = target.codomain.basis_vector(i)
            plus = f(target.codomain.axpy(h, e, target.codomain.copy(p)))
            minus = f(target.codomain.axpy(-h, e, target.codomain.copy(p)))
            assert (plus - minus) / (2.0 * h) == pytest.approx(
                target.codomain.inner_product(gradient, e), rel=1e-5
            )

    def test_push_forward_and_projection(self, setting, rng):
        model, forward, target, truth, data = setting
        feasible = FeasiblePropertySet(forward, target, Ball(model, radius=3.0), data)
        scalar = EuclideanSpace(1)
        row = LinearOperator.from_matrix(
            target.codomain, scalar, np.array([[0.3, -1.0]]), form="galerkin"
        )
        pushed = feasible.push_forward(row)
        e = scalar.basis_vector(0)
        assert pushed.support(e) == pytest.approx(
            feasible.support(row.adjoint(e)), rel=1e-9
        )
        p = target.codomain.random(rng=rng)
        projected = feasible.project(p)
        assert feasible.contains(projected, rtol=1e-6)


class TestTheConversions:
    def test_a_precision_only_ellipsoid_supports_itself_by_conversion(self, rng):
        space = make_dense_metric_space(5)
        precision, covariance = spd_operator(space, rng)
        with_covariance = Ellipsoid(space, precision, covariance=covariance)
        without = Ellipsoid(space, precision)
        assert with_covariance.has_support_function and not without.has_support_function
        q = space.random(rng=rng)
        assert without.support_function()(q) == pytest.approx(
            with_covariance.support_function()(q), rel=1e-7
        )
        point = without.support_maximizer(q)
        assert space.norm(
            space.subtract(point, with_covariance.support_maximizer(q))
        ) <= 1e-5 * space.norm(point)

    def test_a_semidefinite_ellipsoid_is_unbounded_where_its_precision_is_flat(
        self, rng
    ):
        space = EuclideanSpace(4)
        matrix = np.diag([1.0, 2.0, 3.0, 0.0])
        precision = LinearOperator.from_matrix(
            space,
            space,
            matrix,
            form="galerkin",
            traits=Traits.SELF_ADJOINT | Traits.POSITIVE_SEMIDEFINITE,
        )
        cylinder = Ellipsoid(space, precision)
        assert not cylinder.is_bounded
        assert cylinder.contains(np.array([0.0, 0.0, 0.0, 100.0]))
        assert not cylinder.contains(np.array([2.0, 0.0, 0.0, 0.0]))
        flat = space.basis_vector(3)
        assert cylinder.support_function()(flat) == np.inf
        bounded = space.basis_vector(0)
        assert cylinder.support_function()(bounded) == pytest.approx(1.0, rel=1e-6)

    def test_a_support_only_set_projects_by_the_dual(self, rng):
        space = make_dense_metric_space(4)
        ball = Ball(space, radius=1.5, center=space.random(rng=rng))
        oracle = ConvexSet.from_support_function(
            space, ball.support_function(), maximizer=ball.support_maximizer
        )
        assert not oracle.has_projection and not oracle.has_membership
        x = space.scale(3.0, space.random(rng=rng))
        projected = oracle.project(x)
        # The bundle stops on the value, which is quadratic in the error of
        # the argument: a gap of 1e-12 is a projection good to about 1e-6.
        assert space.norm(
            space.subtract(projected, ball.project(x))
        ) <= 1e-5 * space.norm(x)
        assert oracle.contains(ball.project(x), rtol=1e-6)
        assert not oracle.contains(x, rtol=1e-6)
