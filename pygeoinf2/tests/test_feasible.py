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


class TestTheClosedFormThroughTheEstimator:
    """On a ball prior and exact data the estimator's set is the direct one.

    The route this replaced agreed with it to ``1e-10`` on the support, the
    centre and the fitting model.
    """

    def test_support_membership_and_the_centre(self, setting):
        model, forward, target, truth, data = setting
        prior = Ball(model, radius=3.0)
        new = FeasiblePropertySet(forward, target, prior, data)
        old = BackusGilbertParker(LinearForwardProblem(forward), target, prior)(data)
        assert new.form == "closed_form" == old.form
        assert all(names[0] == "closed_form" for names in new.algorithms.values())
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
        ball = Ball(model, radius=3.0)
        # A support function with no maximizer: no algorithm can use it.
        oracle = ConvexSet.from_support_function(model, ball.support_function())
        general = FeasiblePropertySet(
            forward, target, oracle, data, noise=Ball(forward.codomain, radius=0.1)
        )
        assert general.form == "general"
        assert general.algorithms["support"] == ()
        with pytest.raises(NotImplementedError, match="general"):
            general.support(target.codomain.basis_vector(0))
        with pytest.raises(ValueError, match="does not apply"):
            general.support(target.codomain.basis_vector(0), via="closed_form")
        assert not general.has_support_function and not general.has_membership
        quadratic = FeasiblePropertySet(
            forward, target, ball, data, noise=Ball(forward.codomain, radius=0.1)
        )
        assert quadratic.form == "quadratic"
        assert quadratic.algorithms["support"][0] == "primal"
        with pytest.raises(ValueError, match="does not apply"):
            quadratic.support(target.codomain.basis_vector(0), via="closed_form")

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
        assert feasible.structure is sibling.structure

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


# --------------------------------------------------------------------- #
#                           The quadratic tier                           #
# --------------------------------------------------------------------- #


def ellipsoid_on(space, rng, *, center=None, scale=1.0):
    precision, covariance = spd_operator(space, rng, scale=scale)
    return Ellipsoid(space, precision, center=center, covariance=covariance)


class TestThePrimalAgreesWithTheDual:
    """Two balls, an ellipsoid on either side, centres: the primal against
    the dual, with the dense spectrum as an opt-in solver and conjugate
    gradients to within the bracket.

    The bisection route these replaced agreed with the primal to ``1e-8``
    on two balls and ``1e-7`` on ellipsoids with centres, and on the
    inclusion level to ``1e-7`` and ``1e-6``.
    """

    @pytest.fixture
    def noisy(self, setting):
        model, forward, target, truth, data = setting
        return model, forward, target, truth, data, Ball(forward.codomain, radius=0.1)

    def test_two_balls_with_either_solver(self, noisy):
        from pygeoinf2.numerics import CGSolver, SpectralFamilySolver

        model, forward, target, truth, data, noise = noisy
        prior = Ball(model, radius=3.0)
        old = FeasiblePropertySet(forward, target, prior, data, noise=noise)
        for solver in (SpectralFamilySolver(), CGSolver(rtol=1e-12)):
            new = FeasiblePropertySet(
                forward, target, prior, data, noise=noise, solver=solver
            )
            assert new.form == "quadratic"
            assert new.algorithms["support"][0] == "primal"
            assert new.algorithms["inclusion"][0] == "search"
            for q in directions(target.codomain):
                result = new.support_result(q)
                assert result.converged
                assert result.value == pytest.approx(
                    old.support(q, via="dual"), rel=1e-6
                )
                assert result.inner <= result.value + 1e-12
                assert result.gap <= 1e-4 * max(abs(result.value), 1.0)
            p = target(truth)
            assert new.inclusion_level(p) == pytest.approx(
                old.inclusion_level(p), rel=1e-7
            )
            assert new.admits(p, via="distance")
            assert new.contains(p) and not new.is_empty()

    def test_ellipsoids_on_both_sides_off_centre(self, noisy, rng):
        model, forward, target, truth, data, _ = noisy
        prior = ellipsoid_on(
            model, rng, center=model.scale(0.3, truth), scale=1.0 / 9.0
        )
        noise = ellipsoid_on(
            forward.codomain, rng, center=forward.codomain.random(rng=rng), scale=100.0
        )
        # Data that the shifted truth fits: inside both sets by construction.
        shifted_data = forward.codomain.add(forward(truth), noise.center)
        new = FeasiblePropertySet(forward, target, prior, shifted_data, noise=noise)
        assert not new.is_empty()
        for q in directions(target.codomain):
            result = new.support_result(q)
            assert result.value == pytest.approx(new.support(q, via="dual"), rel=1e-6)
            assert result.gap <= 1e-3 * max(abs(result.value), 1.0)
            extremal = result.extremal_model
            assert prior.contains(extremal, rtol=1e-8)
            assert noise.contains(
                forward.codomain.subtract(shifted_data, forward(extremal)), rtol=1e-8
            )
        p = target(truth)
        assert new.admits(p) and new.admits(p, via="distance")

    def test_the_closed_form_is_the_limit_of_vanishing_noise(self, setting):
        model, forward, target, truth, data = setting
        prior = Ball(model, radius=3.0)
        exact = FeasiblePropertySet(forward, target, prior, data)
        tight = FeasiblePropertySet(
            forward, target, prior, data, noise=Ball(forward.codomain, radius=1e-5)
        )
        for q in directions(target.codomain):
            assert tight.support(q) == pytest.approx(exact.support(q), rel=1e-4)

    def test_a_slack_data_constraint_costs_no_solve(self, setting):
        model, forward, target, truth, data = setting
        roomy = Ball(forward.codomain, radius=1e6)
        feasible = FeasiblePropertySet(
            forward, target, Ball(model, radius=3.0), data, noise=roomy
        )
        q = target.codomain.basis_vector(0)
        result = feasible.support_result(q)
        assert result.evaluations == 0 and result.multipliers[1] == 0.0
        assert result.value == pytest.approx(
            3.0 * model.norm(target.adjoint(q)), rel=1e-10
        )
        assert result.gap <= 1e-10


class TestTheSearch:
    def test_feasibility_certifies_and_the_lower_bound_holds(self, setting):
        model, forward, target, truth, data = setting
        noise = Ball(forward.codomain, radius=0.1)
        feasible = FeasiblePropertySet(
            forward, target, Ball(model, radius=3.0), data, noise=noise
        )
        result = feasible.feasibility_result()
        assert result.algorithm == "search" and result.converged
        assert result.lower_bound <= result.level <= feasible.level
        fitting = result.fitting_model
        assert noise.contains(
            forward.codomain.subtract(data, forward(fitting)), rtol=1e-6
        )
        assert model.norm(fitting) ** 2 == pytest.approx(result.level, rel=1e-8)

    def test_an_unfittable_data_set_is_empty(self, setting):
        model, forward, target, truth, data = setting
        feasible = FeasiblePropertySet(
            forward,
            target,
            Ball(model, radius=0.05),
            data,
            noise=Ball(forward.codomain, radius=0.01),
        )
        assert feasible.is_empty()
        assert feasible.feasibility_result().level > feasible.level

    def test_inclusion_fits_the_value_and_the_data(self, setting, rng):
        model, forward, target, truth, data = setting
        noise = Ball(forward.codomain, radius=0.1)
        feasible = FeasiblePropertySet(
            forward, target, Ball(model, radius=3.0), data, noise=noise
        )
        p = target(truth)
        result = feasible.inclusion_result(p)
        assert result.lower_bound <= result.level
        fitting = result.fitting_model
        assert np.allclose(target(fitting), p, atol=1e-8)
        assert noise.contains(
            forward.codomain.subtract(data, forward(fitting)), rtol=1e-6
        )
        assert model.norm(fitting) ** 2 == pytest.approx(result.level, rel=1e-6)
        # A value no model can carry while fitting the data is excluded.
        far = target.codomain.from_components(np.array([50.0, -50.0]))
        assert not feasible.contains(far)
        assert feasible.inclusion_result(far).lower_bound > feasible.level

    def test_the_extent_lies_within_the_bracket(self, setting):
        model, forward, target, truth, data = setting
        scalar = EuclideanSpace(1)
        row = LinearOperator.from_matrix(
            target.codomain, scalar, np.array([[1.0, 0.5]]), form="galerkin"
        )
        feasible = FeasiblePropertySet(
            forward,
            row @ target,
            Ball(model, radius=3.0),
            data,
            noise=Ball(forward.codomain, radius=0.1),
        )
        e = scalar.basis_vector(0)
        lower, upper = feasible.extent(e)
        inner, outer = feasible.support_bracket(e)
        assert inner - 1e-6 <= upper <= outer + 1e-6
        assert upper == pytest.approx(outer, rel=1e-4)


class TestKnobs:
    def test_the_formalisms_agree_and_overdetermined_data_need_the_model_space(
        self, rng
    ):
        from pygeoinf2.geometry.feasible import Primal, Search

        model, data_space, prop = (
            EuclideanSpace(6),
            EuclideanSpace(10),
            EuclideanSpace(2),
        )
        forward = LinearOperator.from_matrix(
            model, data_space, rng.normal(size=(10, 6)), form="galerkin"
        )
        target = LinearOperator.from_matrix(
            model, prop, rng.normal(size=(2, 6)), form="galerkin"
        )
        truth = model.random(rng=rng)
        data = data_space.add(
            forward(truth), data_space.scale(0.05, data_space.random(rng=rng))
        )
        feasible = FeasiblePropertySet(
            forward,
            target,
            Ball(model, radius=3.0 * model.norm(truth)),
            data,
            noise=Ball(data_space, radius=0.5),
        )
        q = prop.basis_vector(0)
        with pytest.raises(ValueError, match="model_space"):
            feasible.support(q)
        by_model = feasible.support_result(q, via=Primal(formalism="model_space"))
        assert by_model.converged
        # The prior is slack: the data alone bound the property, at γ == 0.
        assert by_model.multipliers[0] <= 1e-6 * by_model.multipliers[1]
        # The truth fits the data, so its property is admitted; a value the
        # data pin down away from it is excluded, with a finite certificate.
        admitted = feasible.inclusion_result(
            target(truth), via=Search(formalism="model_space")
        )
        assert admitted.level <= feasible.level * (1.0 + 1e-8)
        excluded = feasible.inclusion_result(
            prop.scale(50.0, prop.basis_vector(1)), via=Search(formalism="model_space")
        )
        assert excluded.lower_bound > feasible.level

    def test_an_instance_with_tuning_is_honoured(self, setting):
        from pygeoinf2.geometry.feasible import Primal

        model, forward, target, truth, data = setting
        feasible = FeasiblePropertySet(
            forward,
            target,
            Ball(model, radius=3.0),
            data,
            noise=Ball(forward.codomain, radius=0.1),
        )
        q = target.codomain.basis_vector(0)
        loose = feasible.support_result(q, via=Primal(tolerance=1e-1))
        tight = feasible.support_result(q, via=Primal(tolerance=1e-6))
        assert loose.gap > tight.gap
        assert loose.value == pytest.approx(tight.value, rel=1e-3)
        with pytest.raises(ValueError, match="does not answer"):
            feasible.inclusion_level(q, via=Primal())

    def test_the_sweep_is_the_same_with_workers(self, setting):
        model, forward, target, truth, data = setting
        feasible = FeasiblePropertySet(
            forward,
            target,
            Ball(model, radius=3.0),
            data,
            noise=Ball(forward.codomain, radius=0.1),
        )
        prop = target.codomain
        angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
        qs = [prop.from_components(np.array([np.cos(a), np.sin(a)])) for a in angles]
        serial = feasible.support_values(qs)
        parallel = feasible.with_data(data).support_values(qs, n_jobs=3)
        assert parallel == pytest.approx(serial, rel=1e-8)
        outer = feasible.polytope(qs)
        inner = feasible.inner_hull(qs)
        assert outer.contains(target(truth)) and not inner.is_outer
        for q in qs:
            assert outer.contains(feasible.support_maximizer(q), rtol=1e-6)

    def test_the_solver_floor_is_checked(self, setting):
        from pygeoinf2.numerics import CGSolver

        model, forward, target, truth, data = setting
        coarse = FeasiblePropertySet(
            forward,
            target,
            Ball(model, radius=3.0),
            data,
            noise=Ball(forward.codomain, radius=1e-9),
            solver=CGSolver(rtol=1e-6),
        )
        with pytest.raises(ValueError, match="floor"):
            coarse.is_empty()


# --------------------------------------------------------------------- #
#                             The general tier                           #
# --------------------------------------------------------------------- #


def quartic_ball(space, radius):
    """``{ ||x||^4 <= r^4 }``: the ball as a sublevel set with a non-quadratic level function."""
    from pygeoinf2.algebra.operators import Functional
    from pygeoinf2.geometry import SublevelSet

    return SublevelSet(
        Functional.from_callables(
            space,
            lambda x: space.squared_norm(x) ** 2,
            gradient=lambda x: space.scale(4.0 * space.squared_norm(x), x),
        ),
        level=radius**4,
    )


def oracle_of(ball):
    return ConvexSet.from_support_function(
        ball.domain, ball.support_function(), maximizer=ball.support_maximizer
    )


class TestTheDual:
    def test_it_agrees_with_the_primal_on_balls_and_the_brackets_overlap(self, setting):
        model, forward, target, truth, data = setting
        noise = Ball(forward.codomain, radius=0.1)
        feasible = FeasiblePropertySet(
            forward, target, Ball(model, radius=3.0), data, noise=noise
        )
        assert feasible.algorithms["support"] == ("primal", "dual", "splitting")
        for q in directions(target.codomain):
            primal = feasible.support_result(q)
            dual = feasible.support_result(q, via="dual")
            assert dual.converged and dual.algorithm == "dual"
            assert dual.value == pytest.approx(primal.value, rel=1e-6)
            assert dual.inner <= dual.value + 1e-9
            assert (
                primal.inner <= dual.value + 1e-9 and dual.inner <= primal.value + 1e-9
            )
            assert dual.certificate is not None
            assert feasible.certificate(q, via="dual") is dual.certificate

    def test_a_quartic_prior_takes_the_dual_and_agrees_with_the_ball(self, setting):
        """The KKT route this replaced agreed to ``1e-4`` on this prior."""
        model, forward, target, truth, data = setting
        prior = quartic_ball(model, 3.0)
        noise = Ball(forward.codomain, radius=0.1)
        feasible = FeasiblePropertySet(forward, target, prior, data, noise=noise)
        assert feasible.form == "general"
        assert feasible.algorithms["support"][0] == "dual"
        old = FeasiblePropertySet(
            forward, target, Ball(model, radius=3.0), data, noise=noise
        )
        for q in directions(target.codomain):
            result = feasible.support_result(q)
            assert result.value == pytest.approx(old.support(q), rel=1e-4)
            extremal = result.extremal_model
            assert model.norm(extremal) <= 3.0 * (1.0 + 1e-4)

    def test_exact_data_with_a_general_prior_is_the_closed_form(self, setting):
        model, forward, target, truth, data = setting
        ball = Ball(model, radius=3.0)
        exact = FeasiblePropertySet(forward, target, ball, data)
        general = FeasiblePropertySet(forward, target, oracle_of(ball), data)
        assert general.form == "general" and general.noise is None
        for q in directions(target.codomain):
            assert general.support(q) == pytest.approx(exact.support(q), rel=1e-6)
        assert not general.is_empty()

    def test_an_empty_set_is_an_unbounded_dual(self, setting):
        model, forward, target, truth, data = setting
        feasible = FeasiblePropertySet(
            forward,
            target,
            oracle_of(Ball(model, radius=0.05)),
            data,
            noise=oracle_of(Ball(forward.codomain, radius=0.01)),
        )
        assert feasible.algorithms["feasibility"] == ("dual",)
        assert feasible.is_empty()
        with pytest.raises(ValueError, match="empty"):
            feasible.support(target.codomain.basis_vector(0))

    def test_the_antipodal_warm_start_and_the_outer_bounds(self, setting):
        model, forward, target, truth, data = setting
        noise = Ball(forward.codomain, radius=0.1)
        feasible = FeasiblePropertySet(
            forward, target, Ball(model, radius=3.0), data, noise=noise
        )
        prop = target.codomain
        angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
        qs = [prop.from_components(np.array([np.cos(a), np.sin(a)])) for a in angles]
        outer = feasible.polytope(qs, via="dual")
        inner = feasible.inner_hull(qs, via="dual")
        assert outer.contains(target(truth)) and not inner.is_outer
        with pytest.raises(ValueError, match="does not apply"):
            feasible.support(qs[0], via="closed_form")


class TestTheSplitting:
    def test_it_agrees_with_the_primal_on_balls(self, setting):
        from pygeoinf2.geometry.feasible import Splitting

        model, forward, target, truth, data = setting
        noise = Ball(forward.codomain, radius=0.1)
        feasible = FeasiblePropertySet(
            forward, target, Ball(model, radius=3.0), data, noise=noise
        )
        splitting = Splitting(tolerance=1e-9, max_iterations=50_000)
        for q in directions(target.codomain)[:2]:
            primal = feasible.support_result(q)
            split = feasible.support_result(q, via=splitting)
            assert split.converged
            assert split.inner == pytest.approx(primal.value, rel=1e-5)
            assert split.value >= split.inner - 1e-9
            assert split.value == pytest.approx(primal.value, rel=1e-4)
            assert noise.contains(
                forward.codomain.subtract(data, forward(split.extremal_model)),
                rtol=1e-6,
            )


class TestDistanceAndTheGeneralSearch:
    def test_distance_agrees_with_the_search_on_balls(self, setting):
        model, forward, target, truth, data = setting
        noise = Ball(forward.codomain, radius=0.1)
        feasible = FeasiblePropertySet(
            forward, target, Ball(model, radius=3.0), data, noise=noise
        )
        assert feasible.algorithms["inclusion"] == ("search", "distance")
        inside = target(truth)
        assert feasible.contains(inside) and feasible.admits(inside, via="distance")
        far = target.codomain.from_components(np.array([5.0, 5.0]))
        assert not feasible.contains(far) and not feasible.admits(far, via="distance")
        result = feasible.inclusion_result(far, via="distance")
        assert result.level == np.inf and "outside" in result.message

    def test_the_general_search_agrees_with_the_ball(self, setting):
        """The likelihood route this replaced agreed to ``1e-5``; the ball's
        level is the squared norm, the quartic's its square."""
        model, forward, target, truth, data = setting
        prior = quartic_ball(model, 3.0)
        noise = Ball(forward.codomain, radius=0.1)
        feasible = FeasiblePropertySet(forward, target, prior, data, noise=noise)
        assert feasible.algorithms["inclusion"] == ("search", "distance")
        assert feasible.algorithms["feasibility"] == ("search", "dual")
        old = FeasiblePropertySet(
            forward, target, Ball(model, radius=3.0), data, noise=noise
        )
        assert not feasible.is_empty()
        p = target(truth)
        result = feasible.inclusion_result(p)
        assert result.level == pytest.approx(old.inclusion_level(p) ** 2, rel=1e-5)
        assert result.lower_bound <= result.level
        assert feasible.contains(p)
        fitting = result.fitting_model
        assert np.allclose(target(fitting), p, atol=1e-6)
        assert noise.contains(
            forward.codomain.subtract(data, forward(fitting)), rtol=1e-6
        )
        # The level side and the support side agree on the scalar property.
        far = target.codomain.from_components(np.array([5.0, 5.0]))
        assert not feasible.contains(far)


class TestTheOuterEllipsoid:
    """The S-procedure ellipsoid on two quadratic sets: certified, and
    checked against the support values direction by direction."""

    @pytest.fixture
    def noisy(self, setting):
        model, forward, target, truth, data = setting
        return model, forward, target, truth, data, Ball(forward.codomain, radius=0.1)

    def test_it_contains_the_set_and_the_formalisms_agree(self, noisy):
        from pygeoinf2.geometry.feasible import SProcedure

        model, forward, target, truth, data, noise = noisy
        feasible = FeasiblePropertySet(
            forward, target, Ball(model, radius=3.0), data, noise=noise
        )
        assert feasible.algorithms["outer_ellipsoid"] == ("s_procedure",)
        result = feasible.outer_ellipsoid_result()
        assert result.converged and result.algorithm == "s_procedure"
        assert 0.0 < result.weight < 1.0
        ellipsoid = result.ellipsoid
        assert ellipsoid.contains(target(truth))
        h = ellipsoid.support_function()
        for q in directions(target.codomain) + [
            target.codomain.from_components(np.array([np.cos(a), np.sin(a)]))
            for a in np.linspace(0.0, np.pi, 7)
        ]:
            assert h(q) >= feasible.support(q) - 1e-9
        other = feasible.outer_ellipsoid_result(via=SProcedure(formalism="model_space"))
        assert other.weight == pytest.approx(result.weight, rel=1e-6)
        assert np.allclose(
            target.codomain.to_components(other.ellipsoid.center),
            target.codomain.to_components(ellipsoid.center),
        )
        # Asked again, nothing is recomputed.
        assert feasible.outer_ellipsoid_result() is result

    def test_each_criterion_wins_on_its_own_measure(self, noisy):
        model, forward, target, truth, data, noise = noisy
        feasible = FeasiblePropertySet(
            forward, target, Ball(model, radius=3.0), data, noise=noise
        )
        space = target.codomain

        def matrix(ellipsoid):
            return np.column_stack(
                [
                    space.to_components(ellipsoid.covariance(space.basis_vector(j)))
                    for j in range(space.dim)
                ]
            )

        by_trace = matrix(feasible.outer_ellipsoid(criterion="trace"))
        by_volume = matrix(feasible.outer_ellipsoid(criterion="volume"))
        assert np.trace(by_trace) <= np.trace(by_volume) * (1.0 + 1e-9)
        assert np.linalg.slogdet(by_volume)[1] <= np.linalg.slogdet(by_trace)[1] + 1e-9
        with pytest.raises(ValueError, match="criterion"):
            feasible.outer_ellipsoid(criterion="area")

    def test_ellipsoids_on_both_sides_off_centre_and_a_metric_property_space(
        self, setting, rng
    ):
        model, forward, _, truth, _ = setting
        property_space = make_dense_metric_space(2)
        target = LinearOperator.from_matrix(
            model, property_space, rng.normal(size=(2, model.dim)), form="galerkin"
        )
        prior = ellipsoid_on(
            model, rng, center=model.scale(0.3, truth), scale=1.0 / 9.0
        )
        noise = ellipsoid_on(
            forward.codomain, rng, center=forward.codomain.random(rng=rng), scale=100.0
        )
        data = forward.codomain.add(forward(truth), noise.center)
        feasible = FeasiblePropertySet(forward, target, prior, data, noise=noise)
        ellipsoid = feasible.outer_ellipsoid(criterion="volume")
        assert ellipsoid.contains(target(truth))
        h = ellipsoid.support_function()
        for _ in range(12):
            q = property_space.random(rng=rng)
            assert h(q) >= feasible.support(q) - 1e-9 * max(1.0, abs(h(q)))

    def test_the_closed_form_is_the_set_itself(self, setting):
        model, forward, target, truth, data = setting
        feasible = FeasiblePropertySet(forward, target, Ball(model, radius=3.0), data)
        assert feasible.algorithms["outer_ellipsoid"] == ("closed_form",)
        result = feasible.outer_ellipsoid_result(criterion="volume")
        assert result.algorithm == "closed_form" and result.weight is None
        space = target.codomain
        assert np.allclose(
            space.to_components(result.ellipsoid.center),
            space.to_components(feasible.ellipsoid.center),
        )
        for q in directions(space):
            assert result.ellipsoid.support_function()(q) == pytest.approx(
                feasible.support(q)
            )

    def test_an_empty_set_and_a_general_set_are_refused(self, setting):
        model, forward, target, truth, data = setting
        far = forward.codomain.scale(50.0, data)
        empty = FeasiblePropertySet(
            forward,
            target,
            Ball(model, radius=3.0),
            far,
            noise=Ball(forward.codomain, radius=0.1),
        )
        with pytest.raises(ValueError, match="empty"):
            empty.outer_ellipsoid()
        general = FeasiblePropertySet(
            forward,
            target,
            quartic_ball(model, 3.0),
            data,
            noise=Ball(forward.codomain, radius=0.1),
        )
        assert general.algorithms["outer_ellipsoid"] == ()
        with pytest.raises(NotImplementedError, match="outer_ellipsoid"):
            general.outer_ellipsoid()
        with pytest.raises(ValueError, match="does not apply"):
            FeasiblePropertySet(
                forward,
                target,
                Ball(model, radius=3.0),
                data,
                noise=Ball(forward.codomain, radius=0.1),
            ).outer_ellipsoid(via="closed_form")
