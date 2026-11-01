"""
The feasible property set in the geometry layer: the closed form, its
generalisation to ellipsoid priors and centres, the dispatch by form, and
the conversions between a set's descriptions.
"""

import numpy as np
import pytest

from pygeoinf.algebra.operators import LinearOperator
from pygeoinf.algebra.spaces import EuclideanSpace
from pygeoinf.geometry import (
    Ball,
    ClosedForm,
    ConvexSet,
    Ellipsoid,
    FeasiblePropertySet,
    InclusionResult,
    LinearCertificate,
    SupportResult,
)
from pygeoinf.inference import BackusGilbertParker, LinearForwardProblem
from pygeoinf.numerics.solvers import ModelSpace
from pygeoinf.traits import Traits

from .conftest import (
    WeightedSpace,
    make_dense_metric_space,
    make_weighted_space,
    spd_matrix,
)


def directions(space):
    return [
        space.scale(sign, space.basis_vector(index))
        for index in range(space.dim)
        for sign in (1.0, -1.0)
    ]


def spd_operator(space, rng, *, scale=1.0):
    """A random positive definite operator and its inverse, metric and all."""
    galerkin = scale * spd_matrix(rng, space.dim)
    traits = Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE
    precision = LinearOperator.from_matrix(
        space, space, galerkin, form="galerkin", traits=traits
    )
    components = np.linalg.solve(space.gram_matrix(), galerkin)
    covariance = LinearOperator.from_matrix(
        space, space, np.linalg.inv(components), form="components", traits=traits
    )
    return precision, covariance


@pytest.fixture(params=["weighted", "dense"])
def setting(request, rng):
    """A weighted model space, an under-determined map, a property, a feasible truth."""
    # A dense Gram matrix on the model space and a diagonal one on the data
    # space, so that every adjoint, precision and dual gradient meets a
    # metric; the weighted case keeps the data space Euclidean.
    if request.param == "dense":
        model = make_dense_metric_space(4)
        data = WeightedSpace(np.array([0.5, 2.0]))
    else:
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

    def test_an_algorithm_answering_on_request_is_reached_by_name_only(self, rng):
        """The linear certificate is listed nowhere as automatic, is refused
        where it cannot build its operator, and answers on quadratic sets
        when named, with an outer value and no inner one."""
        model = make_dense_metric_space(4)
        data_space, property_space = EuclideanSpace(2), EuclideanSpace(2)
        forward = LinearOperator.from_matrix(
            model, data_space, rng.normal(size=(2, 4)), form="galerkin"
        )
        target = LinearOperator.from_matrix(
            model, property_space, rng.normal(size=(2, 4)), form="galerkin"
        )
        truth = model.random(rng=rng)
        feasible = FeasiblePropertySet(
            forward,
            target,
            Ball(model, radius=2.0 * model.norm(truth)),
            forward(truth),
            noise=Ball(data_space, radius=0.1),
        )
        assert feasible.form == "quadratic"
        for probe in ("support", "outer_ellipsoid"):
            assert "linear_certificate" not in feasible.algorithms[probe]
        q = property_space.basis_vector(0)
        assert feasible.support_result(q).algorithm == "primal"
        named = feasible.support_result(q, via="linear_certificate")
        assert named.algorithm == "linear_certificate"
        assert named.inner == -np.inf and named.value >= feasible.support(q) - 1e-9
        instance = LinearCertificate(alpha=2.0)
        assert feasible.support(q, via=instance) >= feasible.support(q) - 1e-9
        assert feasible.outer_ellipsoid(via="linear_certificate").contains(
            target(truth)
        )
        # A general prior: the operator cannot be built, but a given one is taken.
        fat = Ball(model, radius=1.0) + Ball(model, radius=1.0 * model.norm(truth))
        general = FeasiblePropertySet(
            forward, target, fat, forward(truth), noise=Ball(data_space, radius=0.1)
        )
        with pytest.raises(ValueError, match="does not apply"):
            general.support(q, via="linear_certificate")
        given = LinearCertificate(operator=LinearCertificate().operator(feasible))
        assert general.support(q, via=given) >= general.support(q) - 1e-9

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

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(ClosedForm, "_build", counting)
            feasible = FeasiblePropertySet(
                forward, target, Ball(model, radius=3.0), data
            )
            sibling = feasible.with_data(forward.codomain.scale(0.5, data))
            for q in directions(target.codomain):
                feasible.support(q)
                sibling.support(q)
            feasible.contains(target(truth))
            sibling.is_empty()
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


def ellipsoid_on(space, rng, *, center=None, scale=1.0, through=None):
    """A random ellipsoid; with ``through`` rescaled so that point lies inside,
    at level 0.6, whatever the metric made of the random shape."""
    precision, covariance = spd_operator(space, rng, scale=scale)
    if through is not None:
        offset = through if center is None else space.subtract(through, center)
        level = space.inner_product(offset, precision(offset))
        precision = (precision * (0.6 / level)).with_traits(Traits.POSITIVE_DEFINITE)
        covariance = (covariance * (level / 0.6)).with_traits(Traits.POSITIVE_DEFINITE)
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
        from pygeoinf.numerics import CGSolver, SpectralFamilySolver

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
                # The bracket's width follows the primal's tolerance on the
                # multiplier; on a dense model metric it is a few times wider.
                assert result.gap <= 1e-3 * max(abs(result.value), 1.0)
            p = target(truth)
            assert new.inclusion_level(p) == pytest.approx(
                old.inclusion_level(p), rel=1e-7
            )
            assert new.admits(p, via="distance")
            assert new.contains(p) and not new.is_empty()

    def test_ellipsoids_on_both_sides_off_centre(self, noisy, rng):
        model, forward, target, truth, data, _ = noisy
        prior = ellipsoid_on(
            model, rng, center=model.scale(0.3, truth), scale=1.0 / 9.0, through=truth
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


class TestADirectionTheDataCannotSee:
    def test_the_inner_end_reaches_the_prior_boundary(self):
        """The data see two coordinates and the property is the third. The
        exact extent is what the prior ball leaves after the nearest fitting
        model, and the primal's attaining model must reach it even though its
        optimum meets the misfit only to the search's tolerance."""
        model, data_space, prop = (
            EuclideanSpace(3),
            EuclideanSpace(2),
            EuclideanSpace(1),
        )
        forward = LinearOperator.from_matrix(
            model,
            data_space,
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            form="components",
        )
        target = LinearOperator.from_matrix(
            model, prop, np.array([[0.0, 0.0, 1.0]]), form="components"
        )
        e = prop.basis_vector(0)
        for s in np.linspace(0.8, 1.2, 9):
            data = data_space.from_components(s * np.array([1.0, 1.0]))
            feasible = FeasiblePropertySet(
                forward,
                target,
                Ball(model, radius=2.0),
                data,
                noise=Ball(data_space, radius=0.1),
            )
            inner, outer = feasible.support_bracket(e)
            exact = np.sqrt(4.0 - (np.hypot(s, s) - 0.1) ** 2)
            assert outer == pytest.approx(exact, rel=1e-6)
            assert inner == pytest.approx(exact, rel=1e-4)


class TestKnobs:
    def test_a_slack_prior_on_overdetermined_data_needs_the_model_space(self, rng):
        from pygeoinf.geometry.feasible import Primal, Search

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
        by_model = feasible.support_result(q, via=Primal(solver=ModelSpace(None)))
        assert by_model.converged
        # The prior is slack: the data alone bound the property, at γ == 0.
        assert by_model.multipliers[0] <= 1e-6 * by_model.multipliers[1]
        # The data-space family (ten data, six coefficients) is singular
        # there, so its walk stops at the floor and returns a bracket that
        # holds the limit to the order of the floor; nothing refuses by
        # dimension.
        by_data = feasible.support_result(q, via=Primal())
        assert by_data.converged
        assert by_data.inner <= by_model.value + 1e-9 <= by_data.value + 1e-9
        assert by_data.inner == pytest.approx(by_model.value, rel=1e-5)
        assert by_data.value == pytest.approx(by_model.value, rel=1e-2)
        # The truth fits the data, so its property is admitted; a value the
        # data pin down away from it is excluded, with a finite certificate.
        admitted = feasible.inclusion_result(
            target(truth), via=Search(solver=ModelSpace(None))
        )
        assert admitted.level <= feasible.level * (1.0 + 1e-8)
        excluded = feasible.inclusion_result(
            prop.scale(50.0, prop.basis_vector(1)), via=Search(solver=ModelSpace(None))
        )
        assert excluded.lower_bound > feasible.level

    def test_an_instance_with_tuning_is_honoured(self, setting):
        from pygeoinf.geometry.feasible import Primal

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
        # Each worker warm-starts its own root find, so the two agree to the
        # algorithm's accuracy, not to rounding.
        assert parallel == pytest.approx(serial, rel=1e-6)
        outer = feasible.polytope(qs)
        inner = feasible.inner_hull(qs)
        assert outer.contains(target(truth)) and not inner.is_outer
        for q in qs:
            assert outer.contains(feasible.support_maximizer(q), rtol=1e-6)

    def test_the_solver_floor_is_checked(self, setting):
        from pygeoinf.numerics import CGSolver

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
    from pygeoinf.algebra.operators import Functional
    from pygeoinf.geometry import SublevelSet

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
        from pygeoinf.geometry.feasible import Splitting

        model, forward, target, truth, data = setting
        noise = Ball(forward.codomain, radius=0.1)
        feasible = FeasiblePropertySet(
            forward, target, Ball(model, radius=3.0), data, noise=noise
        )
        tolerance = 1e-7
        splitting = Splitting(tolerance=tolerance, max_iterations=50_000)
        for q in directions(target.codomain)[:2]:
            primal = feasible.support_result(q)
            split = feasible.support_result(q, via=splitting)
            assert split.converged
            assert split.inner == pytest.approx(primal.value, rel=1e-5)
            # The iterate is feasible only to the tolerance, so the inner
            # value may sit above the outer bound by that order.
            assert split.value >= split.inner - 10 * tolerance * (1 + abs(primal.value))
            assert split.value == pytest.approx(primal.value, rel=1e-4)
            # Feasible to the tolerance relative to the data's size, so the
            # misfit may pass the noise radius by that much.
            misfit = forward.codomain.norm(
                forward.codomain.subtract(data, forward(split.extremal_model))
            )
            assert misfit <= 0.1 + 10 * tolerance * forward.codomain.norm(data)


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
        from pygeoinf.geometry.feasible import SProcedure

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
        assert result.algorithm == "s_procedure" and "checked" in result.message
        other = feasible.outer_ellipsoid_result(via=SProcedure(solver=None))
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
            model, rng, center=model.scale(0.3, truth), scale=1.0 / 9.0, through=truth
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


class TestABudgetedRunStillCertifies:
    """A run that stops at its budget is still a bracket.

    The outer value is a bound whatever the iterate (weak duality for the
    dual and the splitting, the dual cost at the primal's multiplier for the
    primal), and the inner value is attained by a point of the set; so the
    converged answer lies in ``[inner, value]`` of the unconverged one. That
    is the certificate, and it is what a caller who cannot afford to
    converge is relying on.
    """

    @pytest.fixture
    def balls(self, setting):
        model, forward, target, truth, data = setting
        noise = Ball(forward.codomain, radius=0.1)
        feasible = FeasiblePropertySet(
            forward, target, Ball(model, radius=3.0), data, noise=noise
        )
        return feasible, directions(target.codomain)[:3]

    @pytest.mark.parametrize("route", ["primal", "dual", "splitting"])
    def test_the_converged_answer_lies_in_the_budgeted_bracket(self, balls, route):
        from pygeoinf.geometry.feasible import Dual, Primal, Splitting

        feasible, qs = balls
        algorithm = {
            # The bracketing alone meets the default tolerance here, so the
            # budget bites only against a tolerance below the floor.
            "primal": lambda: Primal(tolerance=1e-14, max_iterations=2),
            "dual": lambda: Dual(max_iterations=2),
            "splitting": lambda: Splitting(tolerance=1e-6, max_iterations=3),
        }[route]()
        for q in qs:
            reference = feasible.support_result(q)
            assert reference.converged
            budgeted = feasible.support_result(q, via=algorithm)
            assert not budgeted.converged, route
            # The attained value is below the truth, which is below the bound;
            # the splitting's point is feasible only to its tolerance.
            slack = 1e-9 * max(1.0, abs(reference.value))
            if route == "splitting":
                slack += 10 * 1e-6 * max(1.0, abs(reference.value))
            assert budgeted.inner <= reference.value + slack, route
            assert budgeted.value >= reference.value - reference.gap - slack, route
            assert budgeted.value >= budgeted.inner - slack, route


class TestTheLinearCertificateOperator:
    def test_it_is_the_regularized_resolution_of_the_property(self, setting, rng):
        """``X == T C A* (A C A* + α R)^-1``, matrix-free, against the same
        expression assembled in components on a dense model metric."""
        from pygeoinf.geometry.feasible import linear_certificate_operator

        model, forward, target, _, _ = setting
        data = forward.codomain
        _, covariance = spd_operator(model, rng)
        _, noise = spd_operator(data, rng, scale=0.1)
        alpha = 0.7
        X = linear_certificate_operator(
            forward, target, covariance=covariance, error_covariance=noise, alpha=alpha
        )
        assert X.domain == data and X.codomain == target.codomain
        A, T = forward.matrix(form="components"), target.matrix(form="components")
        C, R = covariance.matrix(form="components"), noise.matrix(form="components")
        gram_model, gram_data = model.gram_matrix(), data.gram_matrix()
        # Hilbert adjoints in components: A* == G_m^-1 A^T G_d.
        A_star = np.linalg.solve(gram_model, A.T @ gram_data)
        expected = T @ C @ A_star @ np.linalg.inv(A @ C @ A_star + alpha * R)
        d = data.random(rng=rng)
        got = target.codomain.to_components(X(d))
        assert got == pytest.approx(expected @ data.to_components(d), rel=1e-7)


class TestMoreDataThanModel:
    """With more data than model coefficients the data-space family is
    singular at a multiplier of zero and positive definite at every other,
    and the routes probe no other: the feasibility search, the inclusion
    search and the primal all run on the family in the data space, agree
    with the model-space formalism, and report an unfittable data set as
    empty rather than refusing by dimension."""

    @pytest.fixture
    def tall(self, rng):
        model = make_dense_metric_space(3)
        data = WeightedSpace(rng.uniform(0.5, 2.0, 7))
        property_space = EuclideanSpace(2)
        forward = LinearOperator.from_matrix(
            model, data, rng.normal(size=(7, 3)), form="galerkin"
        )
        target = LinearOperator.from_matrix(
            model, property_space, rng.normal(size=(2, 3)), form="galerkin"
        )
        raw = model.random(rng=rng)
        truth = model.scale(1.5 / model.norm(raw), raw)
        noise = Ball(data, radius=0.2)
        small = data.scale(0.05 / data.norm(raw := data.random(rng=rng)), raw)
        return model, forward, target, truth, data.add(forward(truth), small), noise

    def test_the_two_formalisms_agree(self, tall):
        from pygeoinf.geometry.feasible import Primal, Search

        model, forward, target, truth, data, noise = tall
        feasible = FeasiblePropertySet(
            forward, target, Ball(model, radius=3.0), data, noise=noise
        )
        assert not feasible.is_empty(via=Search())
        assert not feasible.is_empty(via=Search(solver=ModelSpace(None)))
        in_data = feasible.feasibility_result(via=Search())
        in_model = feasible.feasibility_result(via=Search(solver=ModelSpace(None)))
        assert in_data.level == pytest.approx(in_model.level, rel=1e-5)
        for q in directions(target.codomain):
            data_side = feasible.support_result(q, via=Primal())
            model_side = feasible.support_result(q, via=Primal(solver=ModelSpace(None)))
            # The inner value is attained either way; the outer one sits
            # above the limit by the order of the primal's floor where the
            # prior is slack.
            assert data_side.inner == pytest.approx(model_side.value, rel=1e-5)
            assert data_side.value == pytest.approx(model_side.value, rel=1e-2)
            assert data_side.inner <= model_side.value + 1e-9 <= data_side.value + 1e-9
        p = target(truth)
        assert feasible.admits(p)
        assert feasible.inclusion_level(p, via=Search()) == (
            pytest.approx(
                feasible.inclusion_level(p, via=Search(solver=ModelSpace(None))),
                rel=1e-5,
            )
        )

    def test_an_unfittable_data_set_is_empty_in_the_data_space(self, tall, rng):
        from pygeoinf.geometry.feasible import Search

        model, forward, target, truth, data, noise = tall
        space = forward.codomain
        far = space.add(data, space.scale(50.0 / space.norm(data), data))
        feasible = FeasiblePropertySet(
            forward, target, Ball(model, radius=0.1), far, noise=noise
        )
        assert feasible.is_empty(via=Search())
        assert feasible.is_empty(via=Search(solver=ModelSpace(None)))


# --------------------------------------------------------------------- #
#                Edge cases of the sets, and what they answer            #
# --------------------------------------------------------------------- #


def problem_on(rng, *, model_dim, data_dim, property_dim):
    """A dense model metric, a weighted data space, random ``A`` and ``T``, a truth of norm two."""
    model = make_dense_metric_space(model_dim)
    data = WeightedSpace(np.linspace(0.5, 2.0, data_dim))
    property_space = EuclideanSpace(property_dim)
    forward = LinearOperator.from_matrix(
        model, data, rng.normal(size=(data_dim, model_dim)), form="galerkin"
    )
    target = LinearOperator.from_matrix(
        model,
        property_space,
        rng.normal(size=(property_dim, model_dim)),
        form="galerkin",
    )
    raw = model.random(rng=rng)
    truth = model.scale(2.0 / model.norm(raw), raw)
    return model, data, property_space, forward, target, truth


def circle(space, count):
    return [
        space.from_components(np.array([np.cos(a), np.sin(a)]))
        for a in np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    ]


class TestTheOuterEllipsoidIsChecked:
    """Twice as many data as model coefficients and a prior from binding to
    slack: the default ellipsoid reaches every attained support value, and
    the data-space formalism, whose shape cancels there, either does too or
    refuses."""

    @pytest.mark.parametrize("radius", [3.0, 10.0, 30.0])
    def test_it_never_falls_below_an_attained_support(self, radius):
        from pygeoinf.geometry.feasible import Primal, SProcedure

        rng = np.random.default_rng(81)
        model, data, prop, forward, target, truth = problem_on(
            rng, model_dim=20, data_dim=40, property_dim=2
        )
        observed = data.add(
            forward(truth), data.scale(0.5 / np.sqrt(40), data.random(rng=rng))
        )
        feasible = FeasiblePropertySet(
            forward,
            target,
            Ball(model, radius=radius),
            observed,
            noise=Ball(data, radius=1.0),
        )
        qs = circle(prop, 8)
        attained = [
            feasible.support_bracket(q, via=Primal(solver=ModelSpace(None)))[0]
            for q in qs
        ]

        def reaches(ellipsoid):
            h = ellipsoid.support_function()
            return all(
                h(q) >= inner - 1e-9 * abs(inner) for q, inner in zip(qs, attained)
            )

        result = feasible.outer_ellipsoid_result()
        assert result.algorithm == "s_procedure" and "checked" in result.message
        assert reaches(result.ellipsoid)
        try:
            by_data = feasible.outer_ellipsoid(via=SProcedure(solver=None))
        except ValueError as error:
            assert "model space" in str(error)
        else:
            assert reaches(by_data)


class TestAFlatSet:
    """Exact data with fewer kernel dimensions than the property: the set is
    a segment, which has support values, membership and a level function,
    and no ellipsoid, projection or outer ellipsoid."""

    def test_it_refuses_what_needs_an_inverse_and_answers_the_rest(self, rng):
        model, data, prop, forward, target, truth = problem_on(
            rng, model_dim=4, data_dim=3, property_dim=2
        )
        feasible = FeasiblePropertySet(
            forward, target, Ball(model, radius=4.0), forward(truth)
        )
        assert feasible.form == "closed_form" and not feasible.has_projection
        for refused in (
            lambda: feasible.ellipsoid,
            lambda: feasible.project(prop.zero()),
            lambda: feasible.outer_ellipsoid(),
        ):
            with pytest.raises(ValueError, match="flat"):
                refused()
        # The segment's two ends are the contact points of a direction.
        q = prop.basis_vector(0)
        first = feasible.support_maximizer(q)
        second = feasible.support_maximizer(prop.negative(q))
        along = prop.subtract(first, second)
        middle = prop.add(second, prop.scale(0.5, along))
        assert feasible.contains(target(truth)) and feasible.contains(middle)
        normal = prop.from_components(np.array([-along[1], along[0]]))
        off = prop.axpy(1e-3 / prop.norm(normal), normal, prop.copy(middle))
        assert not feasible.contains(off)
        # The level's gradient along the segment, against its difference.
        point = prop.add(second, prop.scale(0.3, along))
        unit = prop.scale(1.0 / prop.norm(along), along)
        step = 1e-4 * prop.norm(along)
        slope = (
            feasible.inclusion_level(prop.axpy(step, unit, prop.copy(point)))
            - feasible.inclusion_level(prop.axpy(-step, unit, prop.copy(point)))
        ) / (2.0 * step)
        gradient = feasible.level_function().gradient(point)
        assert prop.inner_product(gradient, unit) == pytest.approx(slope, rel=1e-6)

    def test_a_zero_budget_is_a_single_point(self):
        model, data, prop = EuclideanSpace(3), EuclideanSpace(1), EuclideanSpace(2)
        forward = LinearOperator.from_matrix(
            model, data, np.array([[1.0, 0.0, 0.0]]), form="components"
        )
        target = LinearOperator.from_matrix(
            model, prop, np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]), form="components"
        )
        # The smallest model fitting the datum is e0, of norm one: it uses
        # the whole of the unit ball's budget, and the set is the point T e0.
        feasible = FeasiblePropertySet(
            forward, target, Ball(model, radius=1.0), np.array([1.0])
        )
        point = target(np.array([1.0, 0.0, 0.0]))
        with pytest.raises(ValueError, match="single point"):
            feasible.ellipsoid
        assert feasible.contains(point)
        assert np.allclose(feasible.project(np.array([5.0, -5.0])), point)
        for q in directions(prop):
            assert feasible.support(q) == pytest.approx(prop.inner_product(q, point))


class TestAPointNoiseSetOffTheOrigin:
    def test_its_center_is_an_offset_in_the_data(self, setting, rng):
        model, forward, target, truth, data = setting
        data_space = forward.codomain
        offset = data_space.scale(0.5, data_space.random(rng=rng))
        prior = Ball(model, radius=3.0)
        with_point = FeasiblePropertySet(
            forward,
            target,
            prior,
            data_space.add(data, offset),
            noise=Ball(data_space, radius=0.0, center=offset),
        )
        shifted = FeasiblePropertySet(forward, target, prior, data)
        assert with_point.form == "closed_form" and with_point.noise is None
        for q in directions(target.codomain):
            assert with_point.support(q) == pytest.approx(shifted.support(q), rel=1e-10)
            for route in ("dual", "linear_certificate"):
                assert with_point.support(q, via=route) == pytest.approx(
                    shifted.support(q, via=route), rel=1e-6
                )
        assert np.allclose(with_point.ellipsoid.center, shifted.ellipsoid.center)
        assert with_point.contains(target(truth))
        first = LinearOperator.from_matrix(
            target.codomain,
            EuclideanSpace(1),
            np.array([[1.0, 0.0]]),
            form="components",
        )
        q = EuclideanSpace(1).basis_vector(0)
        assert with_point.push_forward(first).support(q) == pytest.approx(
            shifted.push_forward(first).support(q), rel=1e-10
        )


class TestOutsideWhateverTheSign:
    def test_a_contact_point_is_never_proved_outside(self, setting):
        model, forward, target, truth, data = setting
        far = model.scale(-20.0, truth)
        feasible = FeasiblePropertySet(
            forward, target, Ball(model, radius=3.0, center=far), forward(far)
        )
        qs = directions(target.codomain)
        assert min(feasible.support(q) for q in qs) < 0.0
        for q in qs:
            contact = feasible.support_maximizer(q)
            assert feasible.contains(contact)
            assert not feasible.outside(contact, [q])
            beyond = target.codomain.axpy(1e-3, q, target.codomain.copy(contact))
            assert feasible.outside(beyond, [q])


def shifted_ball(space, shift, *, radius=1.0):
    """The ball written as ``{ ||m||^2 - shift <= radius^2 - shift }``."""
    from pygeoinf.algebra.operators import Functional
    from pygeoinf.geometry import SublevelSet

    return SublevelSet(
        Functional.from_callables(
            space,
            lambda m: space.squared_norm(m) - shift,
            gradient=lambda m: space.scale(2.0, m),
        ),
        level=radius**2 - shift,
    )


class TestTheDualsEmptiness:
    def test_a_prior_written_with_a_negative_level_is_not_empty(self, setting):
        model, forward, target, truth, data = setting
        small = model.scale(0.3 / model.norm(truth), truth)
        noise = oracle_of(Ball(forward.codomain, radius=0.1))
        for shift in (0.0, 2.0):
            feasible = FeasiblePropertySet(
                forward, target, shifted_ball(model, shift), forward(small), noise=noise
            )
            assert feasible.algorithms["feasibility"] == ("dual",)
            assert feasible.level == 1.0 - shift
            assert not feasible.is_empty()
            assert feasible.feasibility_result().level == feasible.level
            with pytest.raises(NotImplementedError, match="exhibiting"):
                feasible.fitting_model()

    def test_a_noise_set_written_with_a_negative_level_fits_the_same(self, setting):
        model, forward, target, truth, data = setting
        prior = quartic_ball(model, 3.0)
        levels = []
        for shift in (0.0, 2.0):
            noise = shifted_ball(forward.codomain, shift, radius=0.1)
            feasible = FeasiblePropertySet(forward, target, prior, data, noise=noise)
            assert feasible.algorithms["feasibility"][0] == "search"
            result = feasible.feasibility_result()
            assert np.isfinite(result.level) and not feasible.is_empty()
            levels.append(result.level)
        assert levels[1] == pytest.approx(levels[0], rel=1e-6)

    def test_a_failing_support_function_is_raised_not_read_as_empty(self, setting):
        model, forward, target, truth, data = setting

        def broken(q):
            raise ValueError("the oracle failed")

        prior = ConvexSet.from_support_function(model, broken, maximizer=broken)
        feasible = FeasiblePropertySet(
            forward,
            target,
            prior,
            data,
            noise=oracle_of(Ball(forward.codomain, radius=0.1)),
        )
        assert feasible.algorithms["feasibility"] == ("dual",)
        with pytest.raises(ValueError, match="the oracle failed"):
            feasible.is_empty()

    @pytest.mark.parametrize("units", [1.0, 1e7])
    def test_a_short_budget_is_not_converged_whatever_the_units(self, rng, units):
        from pygeoinf.geometry.feasible import Dual

        model, data, prop, forward, target, truth = problem_on(
            rng, model_dim=6, data_dim=3, property_dim=2
        )
        center = model.scale(-units, truth)
        prior = Ball(model, radius=0.05 * units, center=center)
        noise = Ball(data, radius=0.01 * units)
        feasible = FeasiblePropertySet(
            forward, target, prior, forward(center), noise=noise
        )
        # The direction in which the property is most negative.
        image = target(center)
        q = prop.scale(-1.0 / prop.norm(image), image)
        short = Dual(max_iterations=3)
        result = feasible.support_result(q, via=short)
        assert result.value < -0.1 * units and not result.converged
        assert result.value >= feasible.support(q) - 1e-9 * units
        assert not feasible.is_empty(via=short)
        # An empty set is still found empty, by a certificate, at any scale.
        far = FeasiblePropertySet(
            forward, target, prior, data.scale(50.0, forward(center)), noise=noise
        )
        assert far.is_empty(via="dual") and far.is_empty()


class TestADegeneratePrior:
    """A semidefinite ellipsoid prior whose flat direction the data see: the
    model-space formalism answers, the data-space one refuses by name."""

    def test_the_model_space_answers_and_the_data_space_names_it(self, rng):
        from pygeoinf.geometry import Ellipsoid
        from pygeoinf.geometry.feasible import Primal, Search

        model, data, prop, forward, target, truth = problem_on(
            rng, model_dim=5, data_dim=3, property_dim=1
        )
        gram = model.gram_matrix()
        flat = rng.normal(size=5)
        semidefinite = gram - np.outer(gram @ flat, gram @ flat) / (flat @ gram @ flat)
        precision = LinearOperator.from_matrix(
            model,
            model,
            semidefinite / 9.0,
            form="galerkin",
            traits=Traits.SELF_ADJOINT | Traits.POSITIVE_SEMIDEFINITE,
        )
        prior = Ellipsoid(model, precision)
        assert not prior.is_bounded
        noise = Ball(data, radius=0.2)
        observed = data.add(forward(truth), data.scale(0.01, data.random(rng=rng)))
        feasible = FeasiblePropertySet(forward, target, prior, observed, noise=noise)
        q = prop.basis_vector(0)
        with pytest.raises(ValueError, match=r"ModelSpace\(None\)"):
            feasible.is_empty()
        with pytest.raises(ValueError, match=r"ModelSpace\(None\)"):
            feasible.support(q)
        assert not feasible.is_empty(via=Search(solver=ModelSpace(None)))
        primal = Primal(solver=ModelSpace(None))
        inner, outer = feasible.support_bracket(q, via=primal)
        assert inner <= outer + 1e-12 and outer - inner <= 1e-4 * abs(outer)
        extremal = feasible.extremal_model(q, via=primal)
        assert prior.contains(extremal, rtol=1e-6)
        assert noise.contains(data.subtract(observed, forward(extremal)), rtol=1e-6)
        # The default outer ellipsoid is in the model space too.
        ellipsoid = feasible.outer_ellipsoid()
        assert ellipsoid.support_function()(q) >= inner


class TestSweepsAndWorkers:
    @pytest.fixture
    def balls(self, setting):
        model, forward, target, truth, data = setting
        feasible = FeasiblePropertySet(
            forward,
            target,
            Ball(model, radius=3.0),
            data,
            noise=Ball(forward.codomain, radius=0.1),
        )
        return feasible, target, truth, data

    def test_a_parallel_sweep_is_kept_by_the_set(self, balls):
        feasible, target, truth, data = balls
        qs = circle(target.codomain, 6)
        sweep = feasible.support_sweep(qs, n_jobs=2)
        for q, result in zip(qs, sweep.results):
            assert feasible.support_result(q) is result

    def test_a_direction_and_its_opposite_share_the_kernel_solve(
        self, balls, monkeypatch
    ):
        from pygeoinf.geometry.feasible import Primal, _QuadraticFamily

        feasible, target, truth, data = balls
        space = feasible.forward_operator.codomain
        cold = []
        original = _QuadraticFamily.solve

        def recording(self, multiplier, right_hand_side, /, *, x0=None):
            if x0 is None:
                cold.append((multiplier, space.to_components(right_hand_side)))
            return original(self, multiplier, right_hand_side, x0=x0)

        monkeypatch.setattr(_QuadraticFamily, "solve", recording)
        primal = Primal()
        for q in directions(target.codomain):
            feasible.support(q, via=primal)
        for i, (multiplier, rhs) in enumerate(cold):
            for earlier, previous in cold[:i]:
                assert not (
                    earlier == multiplier and np.allclose(previous, -rhs, rtol=1e-14)
                )

    def test_the_probes_take_workers_and_agree_with_the_serial_ones(self, balls):
        feasible, target, truth, data = balls
        prop = target.codomain
        qs = circle(prop, 8)
        q = prop.basis_vector(0)
        parallel = feasible.with_data(data)
        assert parallel.extent(q, max_iterations=20, n_jobs=2) == pytest.approx(
            feasible.extent(q, max_iterations=20), rel=1e-9
        )
        serial_hull = feasible.inner_hull(qs)
        parallel_hull = parallel.inner_hull(qs, n_jobs=2)
        # Each worker warm-starts its own root finds, so the contact points
        # agree to the primal's accuracy rather than to rounding.
        serial_points = [prop.to_components(feasible.support_maximizer(p)) for p in qs]
        parallel_points = [
            prop.to_components(parallel.support_maximizer(p)) for p in qs
        ]
        assert np.allclose(parallel_points, serial_points, rtol=1e-3, atol=1e-3)
        for point in parallel_points:
            assert parallel_hull.contains(point, rtol=1e-6)
        for point in serial_points:
            assert serial_hull.contains(point, rtol=1e-6)
        values = [prop.scale(s, target(truth)) for s in (0.0, 0.5, 1.0, 3.0)]
        admitted = feasible.with_data(data).inner_hull(
            values + qs, source="admitted", n_jobs=2
        )
        assert admitted.contains(target(truth), rtol=1e-6)
        ellipsoid = parallel.outer_ellipsoid(n_jobs=2)
        assert np.allclose(
            prop.to_components(ellipsoid.center),
            prop.to_components(feasible.outer_ellipsoid().center),
        )
        far = prop.scale(10.0, prop.from_components(np.array([1.0, 1.0])))
        assert parallel.outside(far, qs, n_jobs=2) and feasible.outside(far, qs)


class TestTheLevelSideOnQuadraticSets:
    def test_the_level_function_has_the_searchs_gradient(self, setting):
        from pygeoinf.geometry.feasible import Search

        model, forward, target, truth, data = setting
        feasible = FeasiblePropertySet(
            forward,
            target,
            Ball(model, radius=3.0),
            data,
            noise=Ball(forward.codomain, radius=0.1),
        )
        assert feasible.form == "quadratic"
        assert feasible.level_function().has_derivative
        assert feasible.has_smooth_level_function
        tight = Search(rtol=1e-12)
        prop = target.codomain
        point = target(truth)
        result = feasible.inclusion_result(point, via=tight)
        # The data constraint binds, so both terms of the gradient count.
        assert result.message == "misfit search converged"
        gradient = tight.level_gradient(feasible, point)
        step = 1e-4 * prop.norm(point)
        for u in (prop.basis_vector(0), prop.basis_vector(1)):
            slope = (
                feasible.inclusion_level(
                    prop.axpy(step, u, prop.copy(point)), via=tight
                )
                - feasible.inclusion_level(
                    prop.axpy(-step, u, prop.copy(point)), via=tight
                )
            ) / (2.0 * step)
            assert prop.inner_product(gradient, u) == pytest.approx(slope, rel=1e-4)

    def test_via_reaches_only_the_probe_it_answers(self, setting):
        model, forward, target, truth, data = setting
        feasible = FeasiblePropertySet(
            forward,
            target,
            Ball(model, radius=3.0),
            data,
            noise=Ball(forward.codomain, radius=0.1),
        )
        # One property, so that the distance's projection is a line search.
        first = LinearOperator.from_matrix(
            target.codomain,
            EuclideanSpace(1),
            np.array([[1.0, 0.0]]),
            form="components",
        )
        line = feasible.push_forward(first)
        q = line.domain.basis_vector(0)
        by_distance = line.extent(q, max_iterations=4, via="distance")
        assert by_distance == pytest.approx(line.extent(q, max_iterations=4), rel=1e-6)
        with pytest.raises(NotImplementedError, match="exhibiting"):
            feasible.fitting_model(via="dual")


def test_the_algorithms_are_exported_with_the_set():
    import pygeoinf.geometry as geometry
    import pygeoinf.geometry.feasible as feasible

    for name in (
        "Algorithm",
        "ClosedForm",
        "Distance",
        "Dual",
        "EllipsoidResult",
        "InclusionResult",
        "LinearCertificate",
        "Primal",
        "SProcedure",
        "Search",
        "Splitting",
        "SupportResult",
        "SweepResult",
    ):
        assert name in feasible.__all__ and name in geometry.__all__, name
        assert getattr(geometry, name) is getattr(feasible, name)
