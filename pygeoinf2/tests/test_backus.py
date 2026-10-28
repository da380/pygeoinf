"""
Set-valued inference: the estimators over the feasible property set.

The closed form, the linear certificate and the iterative algorithms compute
or bound the same object, so every test here is a comparison rather than an
assertion. The strongest cross a method boundary: the primal against the
closed form as the noise vanishes, the dual against the primal, and the
level side against the support side on every point both have an opinion on.

The routes these algorithms replaced set the bars recorded here: the
bisection route agreed with the primal to ``1e-8`` on two balls and ``1e-7``
on ellipsoids with centres; the KKT route with the dual to ``1e-4`` on a
quartic prior; the likelihood route with the search to ``1e-5``.
"""

import numpy as np
import pytest

from pygeoinf2.algebra.operators import Functional, LinearOperator
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.geometry import LinearCertificate, SublevelSet
from pygeoinf2.geometry.convex import Ball, ConvexSet, Ellipsoid, HalfSpace, Polytope
from pygeoinf2.geometry.feasible import Dual
from pygeoinf2.inference import (
    BackusGilbert,
    BackusGilbertParker,
    FeasiblePropertySet,
    ForwardProblem,
    LinearForwardProblem,
    LinearizedFeasiblePropertySet,
    LinearPointEstimator,
)
from pygeoinf2.inference.backus import harden_error
from pygeoinf2.probability.gaussian import GaussianMeasure
from pygeoinf2.symmetric_space.plane import Sobolev as PlaneSobolev
from pygeoinf2.traits import Traits

from .conftest import WeightedSpace, make_dense_metric_space, make_weighted_space


@pytest.fixture(params=["weighted", "dense"])
def setting(request, rng):
    """A model space, an under-determined map, a property, and a feasible truth."""
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
    truth = model.scale(2.0 / model.norm(raw), raw)  # one vector, scaled
    return model, forward, target, truth, forward(truth)


def directions(space):
    return [
        space.scale(sign, space.basis_vector(index))
        for index in range(space.dim)
        for sign in (1.0, -1.0)
    ]


def quartic_ball(space, radius):
    """``||x||^4 <= r^4``: the ball as a sublevel set with a non-quadratic level function."""
    return SublevelSet(
        Functional.from_callables(
            space,
            lambda x: space.squared_norm(x) ** 2,
            gradient=lambda x: space.scale(4.0 * space.squared_norm(x), x),
        ),
        level=radius**4,
    )


def anisotropic_ellipsoid(model, scales, *, center=None):
    """An ellipsoid whose precision has the given Galerkin diagonal.

    Self-adjoint on any metric: a Galerkin matrix is symmetric or the
    operator is not self-adjoint, and ``G diag(s)`` is symmetric only on a
    diagonal ``G``. The covariance is the exact inverse, ``G diag(1/s) G``
    in the Galerkin form.
    """
    gram = model.gram_matrix()
    scale = np.diag(np.asarray(scales)[: model.dim])
    precision = LinearOperator.from_matrix(
        model,
        model,
        np.linalg.inv(scale),
        traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE,
        form="galerkin",
    )
    covariance = LinearOperator.from_matrix(
        model,
        model,
        gram @ scale @ gram,
        traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE,
        form="galerkin",
    )
    return Ellipsoid(model, precision, center=center, covariance=covariance)


def box_prior(model):
    return Polytope(
        model,
        [HalfSpace(model, model.basis_vector(i), offset=1.0) for i in range(2)],
        outer=True,
    )


class TestClosedForm:
    def test_the_truth_is_in_the_set(self, setting):
        model, forward, target, truth, data = setting
        assert model.norm(truth) <= 3.0
        inference = BackusGilbertParker(
            LinearForwardProblem(forward), target, Ball(model, radius=3.0)
        )
        assert inference.form == "closed_form"
        assert inference(data).contains(target(truth))

    def test_the_center_is_the_minimum_norm_property(self, setting):
        model, forward, target, truth, data = setting
        inference = BackusGilbertParker(
            LinearForwardProblem(forward), target, Ball(model, radius=3.0)
        )
        assert np.allclose(
            inference(data).ellipsoid.center, target(inference(data).fitting_model())
        )

    def test_every_feasible_model_lands_inside(self, setting, rng):
        """Sampled from the feasible set itself, so a miss would be a defect."""
        from pygeoinf2.geometry.subspaces import OrthogonalProjector

        model, forward, target, truth, data = setting
        inference = BackusGilbertParker(
            LinearForwardProblem(forward), target, Ball(model, radius=3.0)
        )
        answer = inference(data)
        anchor = answer.fitting_model()
        kernel = OrthogonalProjector.onto_kernel(forward)
        for _ in range(300):
            offset = kernel(model.random(rng=rng))
            candidate = model.axpy(rng.uniform(0.0, 3.0), offset, model.copy(anchor))
            if model.norm(candidate) <= 3.0:
                assert answer.contains(target(candidate))

    def test_the_prior_alone_brackets_the_answer(self, setting):
        model, forward, target, truth, data = setting
        inference = BackusGilbertParker(
            LinearForwardProblem(forward), target, Ball(model, radius=3.0)
        )
        answer = inference(data)
        for direction in directions(inference.property_space):
            assert 3.0 * model.norm(target.adjoint(direction)) >= (
                answer.support(direction) - 1e-8
            )

    def test_data_the_prior_cannot_fit_are_refused(self, setting):
        model, forward, target, truth, data = setting
        inference = BackusGilbertParker(
            LinearForwardProblem(forward), target, Ball(model, radius=0.01)
        )
        feasible = inference(data)
        assert feasible.is_empty()
        with pytest.raises(ValueError, match="fits these data"):
            feasible.ellipsoid
        with pytest.raises(ValueError, match="empty"):
            feasible.support(target.codomain.basis_vector(0))

    def test_the_structure_is_shared_across_data(self, setting, rng):
        """Only the center and the size depend on the data -- the same shape
        as a Gaussian estimator's data-independent covariance."""
        model, forward, target, truth, data = setting
        inference = BackusGilbertParker(
            LinearForwardProblem(forward), target, Ball(model, radius=3.0)
        )
        other = forward(model.scale(0.5, truth))
        assert inference(data).structure is inference(other).structure
        assert inference(data).structure is inference.structure


class TestInclusionTest:
    def test_it_agrees_with_the_closed_form_everywhere(self, setting, rng):
        """Two entirely different computations of one statement.

        The set comes from a projection and an eigen-shape; the test comes
        from a minimum-norm solve on the joint map. They must agree on every
        candidate, and a disagreement would name which is wrong.
        """
        model, forward, target, truth, data = setting
        inference = BackusGilbertParker(
            LinearForwardProblem(forward), target, Ball(model, radius=3.0)
        )
        answer = inference(data)
        ellipsoid = answer.ellipsoid
        space = inference.property_space
        inside = 0
        for _ in range(200):
            candidate = space.from_components(
                space.to_components(ellipsoid.center) + rng.normal(size=space.dim) * 1.5
            )
            by_set = ellipsoid.contains(candidate)
            inside += by_set
            assert answer.admits(candidate) == by_set
        assert 20 < inside < 180  # the sample straddles the boundary

    def test_the_truth_is_admitted_at_its_own_level(self, setting):
        model, forward, target, truth, data = setting
        inference = BackusGilbertParker(
            LinearForwardProblem(forward), target, Ball(model, radius=3.0)
        )
        answer = inference(data)
        assert answer.inclusion_level(target(truth)) <= model.squared_norm(truth) + 1e-8
        assert answer.admits(target(truth))


class TestPrimal:
    @pytest.mark.parametrize("noise", [1e-3, 1e-6])
    def test_it_agrees_with_the_closed_form_as_the_noise_vanishes(self, setting, noise):
        """The parity test between the primal and the closed form.

        Two methods with nothing in common, a one-multiplier minimization
        against a projection and an eigen-shape, converging on the same
        numbers as the noise ball shrinks. The agreement tracks the noise
        radius, which is what says the difference is the problem and not
        the method.
        """
        model, forward, target, truth, data = setting
        problem = LinearForwardProblem(forward)
        prior = Ball(model, radius=3.0)
        exact = BackusGilbertParker(problem, target, prior)(data).support_function()
        primal = BackusGilbertParker(
            problem, target, prior, noise=Ball(forward.codomain, radius=noise)
        )
        assert primal.form == "quadratic"
        for direction in directions(target.codomain):
            assert primal(data).support(direction) == pytest.approx(
                exact(direction), rel=10.0 * noise
            )

    def test_the_extremal_model_saturates_the_prior(self, setting):
        """Both constraints are active at the optimum, so the norm is the
        radius to within the bracket: the model reported is the furthest
        feasible point along the ray, and never outside."""
        model, forward, target, truth, data = setting
        problem = LinearForwardProblem(forward)
        primal = BackusGilbertParker(
            problem,
            target,
            Ball(model, radius=3.0),
            noise=Ball(forward.codomain, radius=1e-4),
        )
        for direction in directions(target.codomain):
            extremal = primal(data).extremal_model(direction)
            assert model.norm(extremal) <= 3.0 * (1.0 + 1e-10)
            assert model.norm(extremal) == pytest.approx(3.0, rel=1e-3)

    def test_the_extremal_model_is_feasible_and_attains_the_inner_value(
        self, setting, rng
    ):
        model, forward, target, truth, data = setting
        noise = Ball(forward.codomain, radius=0.2)
        problem = LinearForwardProblem(forward, error=noise)
        primal = BackusGilbertParker(problem, target, Ball(model, radius=3.0))
        for direction in directions(target.codomain):
            result = primal(data).support_result(direction)
            extremal = result.extremal_model
            assert model.norm(extremal) <= 3.0 + 1e-6
            residual = forward.codomain.subtract(data, forward(extremal))
            assert forward.codomain.norm(residual) <= 0.2 + 1e-6
            assert target.codomain.inner_product(
                target(extremal), direction
            ) == pytest.approx(result.inner)
            assert result.inner <= result.value
            assert result.value == pytest.approx(result.inner, rel=1e-3)

    def test_a_slack_data_constraint_gives_the_prior_bound(self, setting):
        """When the prior's own support point already fits, there is nothing
        to solve and the answer is ``M ||T* q||``."""
        model, forward, target, truth, data = setting
        problem = LinearForwardProblem(
            forward, error=Ball(forward.codomain, radius=1e6)
        )
        primal = BackusGilbertParker(problem, target, Ball(model, radius=3.0))
        direction = target.codomain.basis_vector(0)
        assert primal(data).support(direction) == pytest.approx(
            3.0 * model.norm(target.adjoint(direction)), rel=1e-6
        )


def dense_setting(rng, *, dim=5):
    """The same shape of problem on a space whose Gram matrix is dense."""
    model = make_dense_metric_space(dim)
    data = EuclideanSpace(3)
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


class TestLinearCertificate:
    """The feasible set's ``linear_certificate`` route: an outer bound from a
    fixed weighting of the data, valid for any sets, never chosen unasked."""

    def test_it_bounds_the_exact_set_and_is_asked_for_by_name(self, setting):
        """Validity is free; only sharpness is lost. That is weak duality, and
        it is the property that makes the linear certificate safe to use."""
        model, forward, target, truth, data = setting
        noise = Ball(forward.codomain, radius=0.2)
        problem = LinearForwardProblem(forward, error=noise)
        feasible = BackusGilbertParker(problem, target, Ball(model, radius=3.0))(data)
        assert "linear_certificate" not in feasible.algorithms["support"]
        assert "linear_certificate" not in feasible.algorithms["outer_ellipsoid"]
        operator = LinearCertificate().operator(feasible)
        for direction in directions(target.codomain):
            exact = feasible.support_result(direction)
            assert exact.algorithm != "linear_certificate"
            result = feasible.support_result(direction, via="linear_certificate")
            assert result.algorithm == "linear_certificate"
            assert result.value >= exact.value - 1e-6
            assert result.inner == -np.inf and result.converged
            assert np.allclose(
                forward.codomain.to_components(result.certificate),
                forward.codomain.to_components(operator.adjoint(direction)),
            )
            assert feasible.support_bracket(direction, via="linear_certificate") == (
                -np.inf,
                result.value,
            )
        box = feasible.polytope(directions(target.codomain), via="linear_certificate")
        assert box.contains(target(truth))

    def test_the_bound_holds_on_a_dense_metric(self, rng):
        model, forward, target, truth, data = dense_setting(rng)
        problem = LinearForwardProblem(
            forward, error=Ball(forward.codomain, radius=0.2)
        )
        feasible = BackusGilbertParker(problem, target, Ball(model, radius=3.0))(data)
        route = LinearCertificate()
        for direction in directions(target.codomain):
            value = feasible.support(direction, via=route)
            assert value >= feasible.support(direction) - 1e-6
            assert value == pytest.approx(sum(route.terms(feasible, direction)))
        assert feasible.outer_ellipsoid(via=route).contains(target(truth))

    def test_on_quadratic_sets_the_sum_has_one_ellipsoid(self, setting):
        model, forward, target, truth, data = setting
        noise = Ball(forward.codomain, radius=0.2)
        problem = LinearForwardProblem(forward, error=noise)
        prior = Ball(model, radius=3.0)
        feasible = BackusGilbertParker(problem, target, prior)(data)
        space = target.codomain
        route = LinearCertificate()
        for criterion in ("trace", "volume"):
            result = feasible.outer_ellipsoid_result(criterion=criterion, via=route)
            assert result.algorithm == "linear_certificate" and result.converged
            ellipsoid = result.ellipsoid
            # The ellipsoid sits about the estimate: the first term of the
            # support value along each axis.
            assert np.allclose(
                space.to_components(ellipsoid.center),
                [
                    route.terms(feasible, space.basis_vector(i))[0]
                    for i in range(space.dim)
                ],
            )
            assert ellipsoid.contains(target(truth))
            for direction in directions(space):
                assert ellipsoid.support_function()(direction) >= (
                    feasible.support(direction, via=route) - 1e-9
                )
        # A certificate evaluated against a general set has no ellipsoid.
        general = BackusGilbertParker(
            problem, target, Ball(model, radius=2.0) + Ball(model, radius=1.0)
        )(data)
        given = LinearCertificate(operator=route.operator(feasible))
        with pytest.raises(NotImplementedError, match="ball or an ellipsoid"):
            general.outer_ellipsoid(via=given)

    def test_the_terms_split_the_width_into_two_causes(self, setting):
        model, forward, target, truth, data = setting
        problem = LinearForwardProblem(
            forward, error=Ball(forward.codomain, radius=0.05)
        )
        feasible = BackusGilbertParker(problem, target, Ball(model, radius=3.0))(data)
        route = LinearCertificate()
        space = target.codomain
        for index in range(space.dim):
            q = space.basis_vector(index)
            estimate, resolution, noise = route.terms(feasible, q)
            assert resolution > 0.0 and noise > 0.0
            assert feasible.support(q, via=route) == pytest.approx(
                estimate + resolution + noise
            )
            assert abs(target(truth)[index] - estimate) <= resolution + noise + 1e-8

    def test_less_noise_narrows_only_the_noise_term(self, setting):
        """With the weighting held fixed, the resolution term is a statement
        about the prior set alone."""
        model, forward, target, truth, data = setting
        prior = Ball(model, radius=3.0)
        wide = BackusGilbertParker(
            LinearForwardProblem(forward),
            target,
            prior,
            noise=Ball(forward.codomain, radius=0.5),
        )(data)
        narrow = BackusGilbertParker(
            LinearForwardProblem(forward),
            target,
            prior,
            noise=Ball(forward.codomain, radius=0.05),
        )(data)
        route = LinearCertificate(operator=LinearCertificate().operator(wide))
        for q in directions(target.codomain):
            _, resolution_wide, noise_wide = route.terms(wide, q)
            _, resolution_narrow, noise_narrow = route.terms(narrow, q)
            assert resolution_narrow == pytest.approx(resolution_wide)
            assert noise_narrow < noise_wide

    def test_the_classical_formula_on_two_balls(self, rng):
        """``T A* (A A* + (D/M)² I)⁻¹``, on Euclidean spaces where the
        components are the operator; the route builds the same ``X`` the
        point estimator does."""
        model, data_space, prop = (
            EuclideanSpace(6),
            EuclideanSpace(3),
            EuclideanSpace(2),
        )
        a = rng.normal(size=(3, 6))
        t = rng.normal(size=(2, 6))
        forward = LinearOperator.from_matrix(model, data_space, a, form="components")
        target = LinearOperator.from_matrix(model, prop, t, form="components")
        radius, noise = 2.0, 0.3
        problem = LinearForwardProblem(forward, error=Ball(data_space, radius=noise))
        estimator = BackusGilbert(
            problem, target, covariance=Ball(model, radius=radius).covariance
        )
        classical = t @ a.T @ np.linalg.inv(a @ a.T + (noise / radius) ** 2 * np.eye(3))
        assert np.allclose(estimator.operator.matrix(form="components"), classical)
        data = data_space.random(rng=rng)
        assert np.allclose(estimator(data), classical @ data)
        feasible = BackusGilbertParker(problem, target, Ball(model, radius=radius))(
            data
        )
        built = LinearCertificate().operator(feasible)
        assert np.allclose(built.matrix(form="components"), classical)

    def test_centres_are_carried_through(self, setting):
        """An off-centre prior: the estimate moves by ``(T − X A) a`` and the
        set's reach about it is the centred ball's."""
        model, forward, target, truth, data = setting
        noise = Ball(forward.codomain, radius=0.1)
        problem = LinearForwardProblem(forward, error=noise)
        center = model.scale(0.3, truth)
        centred = BackusGilbertParker(problem, target, Ball(model, radius=3.0))(data)
        shifted = BackusGilbertParker(
            problem, target, Ball(model, radius=3.0, center=center)
        )(data)
        route = LinearCertificate()
        operator = route.operator(centred)
        assert np.allclose(
            operator.matrix(form="components"),
            route.operator(shifted).matrix(form="components"),
        )
        unresolved = target - operator @ forward
        space = target.codomain
        for q in directions(space):
            estimate, resolution, noise_term = route.terms(shifted, q)
            assert estimate == pytest.approx(
                route.terms(centred, q)[0] + space.inner_product(q, unresolved(center))
            )
            assert resolution == pytest.approx(3.0 * model.norm(unresolved.adjoint(q)))
            assert noise_term == pytest.approx(
                0.1 * forward.codomain.norm(operator.adjoint(q))
            )

    def test_a_general_prior_takes_a_given_operator(self, setting):
        """Building ``X`` needs quadratic sets; any convex set with a support
        function can be evaluated against an operator the caller gives, and
        the set it certifies is valid."""
        model, forward, target, truth, data = setting
        noise = Ball(forward.codomain, radius=0.1)
        problem = LinearForwardProblem(forward, error=noise)
        fat = Ball(model, radius=2.0) + Ball(model, radius=1.0)
        ball = BackusGilbertParker(problem, target, Ball(model, radius=3.0))(data)
        general = BackusGilbertParker(problem, target, fat)(data)
        with pytest.raises(ValueError, match="does not apply"):
            general.support(target.codomain.basis_vector(0), via="linear_certificate")
        with pytest.raises(ValueError, match="does not apply"):
            BackusGilbertParker(problem, target, box_prior(model))(data).support(
                target.codomain.basis_vector(0), via=LinearCertificate()
            )
        given = LinearCertificate(operator=LinearCertificate().operator(ball))
        for q in directions(target.codomain):
            # The sum of two balls is the ball of the summed radii, so the
            # evaluated bound is the ball's.
            assert general.support(q, via=given) == pytest.approx(
                ball.support(q, via="linear_certificate")
            )
            assert general.support(q, via=given) >= general.support(q) - 1e-6
        with pytest.raises(ValueError, match="does not apply"):
            general.support(
                target.codomain.basis_vector(0), via=LinearCertificate(operator=target)
            )

    def test_an_ellipsoidal_prior_is_taken(self, setting):
        model, forward, target, truth, data = setting
        prior = anisotropic_ellipsoid(model, [36.0, 16.0, 16.0, 9.0])
        assert prior.contains(truth)
        problem = LinearForwardProblem(
            forward, error=Ball(forward.codomain, radius=0.1)
        )
        feasible = BackusGilbertParker(problem, target, prior)(data)
        for q in directions(target.codomain):
            assert feasible.support(q, via="linear_certificate") >= (
                feasible.support(q) - 1e-6
            )
        assert feasible.outer_ellipsoid(via="linear_certificate").contains(
            target(truth)
        )

    def test_a_nonsense_trade_off_is_refused(self, setting):
        model, forward, target, truth, data = setting
        with pytest.raises(ValueError, match="alpha"):
            LinearCertificate(alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            BackusGilbert(LinearForwardProblem(forward), target, alpha=0.0)


class TestBackusGilbert:
    """The averaging-kernel estimate: a point estimator with its budget."""

    def test_it_is_a_point_estimator_and_an_operator(self, rng):
        model, forward, target, truth, data = dense_setting(rng)
        error = GaussianMeasure.from_standard_deviation(forward.codomain, 0.1)
        problem = LinearForwardProblem(forward, error=error)
        estimator = BackusGilbert(problem, target, alpha=0.5)
        assert isinstance(estimator, LinearPointEstimator)
        assert estimator.alpha == 0.5
        assert estimator.property_operator is target
        space = target.codomain
        assert np.allclose(estimator(data), estimator.operator(data))
        x = model.random(rng=rng)
        assert np.allclose(
            estimator.unresolved(x),
            space.subtract(target(x), estimator.resolution(x)),
        )
        # The budget from the base: the propagated covariance is X R X*.
        propagated = estimator.propagated_covariance()
        q = space.basis_vector(0)
        assert space.inner_product(propagated(q), q) == pytest.approx(
            0.01 * forward.codomain.squared_norm(estimator.operator.adjoint(q))
        )
        # The identity is the default weighting.
        assert np.allclose(
            model.to_components(estimator.covariance(x)), model.to_components(x)
        )
        assert np.allclose(
            estimator.bias(truth), estimator.unresolved(model.scale(-1.0, truth))
        )

    def test_the_same_kernels_certify_a_set_through_the_route(self, setting):
        model, forward, target, truth, data = setting
        noise = Ball(forward.codomain, radius=0.1)
        problem = LinearForwardProblem(forward, error=noise)
        prior = Ball(model, radius=3.0)
        estimator = BackusGilbert(problem, target, covariance=prior.covariance)
        feasible = BackusGilbertParker(problem, target, prior)(data)
        route = LinearCertificate(operator=estimator.operator)
        space = target.codomain
        for q in directions(space):
            certified = feasible.support(q, via=route)
            assert certified == pytest.approx(
                space.inner_product(q, estimator(data))
                + 3.0 * model.norm(estimator.unresolved.adjoint(q))
                + 0.1 * forward.codomain.norm(estimator.operator.adjoint(q))
            )
            assert certified == pytest.approx(
                feasible.support(q, via="linear_certificate")
            )
            assert certified >= feasible.support(q) - 1e-6

    def test_the_error_covariance_is_read_from_the_problem(self, setting, rng):
        model, forward, target, truth, data = setting
        data_space = forward.codomain
        gaussian = GaussianMeasure.from_standard_deviation(data_space, 0.3)
        from_measure = BackusGilbert(
            LinearForwardProblem(forward, error=gaussian), target
        )
        from_ball = BackusGilbert(
            LinearForwardProblem(forward, error=Ball(data_space, radius=0.3)), target
        )
        assert np.allclose(
            from_measure.operator.matrix(form="components"),
            from_ball.operator.matrix(form="components"),
        )
        exact = BackusGilbert(LinearForwardProblem(forward), target)
        assert exact.error_covariance is None
        # Without a noise penalty the kernels reproduce T on the range of A*.
        pulled = forward.adjoint(data_space.random(rng=rng))
        assert np.allclose(exact.resolution(pulled), target(pulled), atol=1e-8)
        with pytest.raises(ValueError, match="data error measure"):
            from_ball.propagated_covariance()
        fat = Ball(data_space, radius=0.2) + Ball(data_space, radius=0.1)
        with pytest.raises(TypeError, match="error_covariance"):
            BackusGilbert(LinearForwardProblem(forward, error=fat), target)
        given = BackusGilbert(
            LinearForwardProblem(forward, error=fat),
            target,
            error_covariance=Ball(data_space, radius=0.3).covariance,
        )
        assert np.allclose(
            given.operator.matrix(form="components"),
            from_ball.operator.matrix(form="components"),
        )

    def test_push_forward_composes_the_kernels(self, setting):
        model, forward, target, truth, data = setting
        problem = LinearForwardProblem(
            forward, error=Ball(forward.codomain, radius=0.1)
        )
        estimator = BackusGilbert(problem, target)
        summary = LinearOperator.from_matrix(
            target.codomain,
            EuclideanSpace(1),
            np.ones((1, target.codomain.dim)),
            form="components",
        )
        pushed = estimator.push_forward(summary)
        assert isinstance(pushed, BackusGilbert)
        assert np.allclose(pushed(data), summary(estimator(data)))
        q = EuclideanSpace(1).basis_vector(0)
        assert np.allclose(
            pushed.unresolved.adjoint(q),
            estimator.unresolved.adjoint(summary.adjoint(q)),
        )


class TestTheLinearizedSet:
    """A nonlinear forward problem linearized about a model: exact for the
    linearized problem, a local statement about the real one."""

    def test_a_linear_problem_linearized_anywhere_gives_the_linear_set(self, rng):
        model, forward, target, truth, data = dense_setting(rng)
        problem = LinearForwardProblem(
            forward, error=Ball(forward.codomain, radius=0.2)
        )
        prior = Ball(model, radius=3.0, center=model.scale(0.3, truth))
        plain = BackusGilbertParker(problem, target, prior)(data)
        anywhere = model.random(rng=rng)
        estimator = BackusGilbertParker(problem, target, prior, model=anywhere)
        linearized = estimator(data)
        assert isinstance(linearized, LinearizedFeasiblePropertySet)
        assert not isinstance(plain, LinearizedFeasiblePropertySet)
        assert linearized.form == plain.form == "quadratic"
        assert linearized.linearization_point is anywhere
        assert np.allclose(linearized.linearization_property, target(anywhere))
        assert linearized.observed_data is data
        assert np.allclose(linearized.data, data)
        for q in directions(target.codomain):
            assert linearized.support_bracket(q) == pytest.approx(
                plain.support_bracket(q)
            )
            assert linearized.support(q, via="linear_certificate") == pytest.approx(
                plain.support(q, via="linear_certificate")
            )
        assert linearized.contains(target(truth))
        assert (
            model.norm(
                model.subtract(linearized.fitting_model(), plain.fitting_model())
            )
            < 1e-8
        )
        # The same through a linearization in hand, and through the pushes.
        in_hand = BackusGilbertParker(
            problem, target, prior, linearization=forward.at(anywhere)
        )(data)
        q = target.codomain.basis_vector(0)
        assert in_hand.support(q) == pytest.approx(plain.support(q))
        summary = LinearOperator.from_matrix(
            target.codomain, EuclideanSpace(1), np.ones((1, 2)), form="components"
        )
        for pushed in (
            linearized.push_forward(summary),
            estimator.push_forward(summary)(data),
            linearized.with_data(data).push_forward(summary),
        ):
            assert isinstance(pushed, LinearizedFeasiblePropertySet)
            assert pushed.support(EuclideanSpace(1).basis_vector(0)) == pytest.approx(
                plain.support(summary.adjoint(EuclideanSpace(1).basis_vector(0)))
            )
        with pytest.raises(ValueError, match="not both"):
            BackusGilbertParker(
                problem,
                target,
                prior,
                model=anywhere,
                linearization=forward.at(anywhere),
            )

    def test_the_ray_problem_contains_the_linearization_points_own_property(self, rng):
        space = PlaneSobolev((24, 24), 2.0, 0.15, bounds=[(0.0, 2.0), (0.0, 2.0)])
        paths = [
            (
                np.array([0.3 + 0.5 * rng.uniform(), 0.3 + 0.5 * rng.uniform()]),
                np.array([1.2 + 0.5 * rng.uniform(), 1.2 + 0.5 * rng.uniform()]),
            )
            for _ in range(5)
        ]
        operator = space.ray_travel_time_operator(paths, 2.0, count=10)
        raw = space.heat_measure(0.3 * space.length_scale).sample(rng=rng)
        point = space.scale(0.2 / np.max(np.abs(space.grid_values(raw))), raw)
        data = operator(point)
        target = space.geodesic_ball_average_operator(
            [np.array([1.0, 1.0]), np.array([0.6, 1.3])], 0.3
        )
        problem = ForwardProblem(
            operator,
            error=Ball(operator.codomain, radius=1e-3 * np.linalg.norm(data)),
        )
        prior = Ball(space, radius=2.0 * space.norm(point))
        with pytest.raises(TypeError, match="nonlinear"):
            BackusGilbertParker(problem, target, prior)
        feasible = BackusGilbertParker(problem, target, prior, model=point)(data)
        assert isinstance(feasible, LinearizedFeasiblePropertySet)
        own = feasible.linearization_property
        assert np.allclose(own, target(point))
        assert feasible.contains(own)
        P = target.codomain
        for index in range(P.dim):
            q = P.basis_vector(index)
            lower, upper = -feasible.support(P.scale(-1.0, q)), feasible.support(q)
            assert lower - 1e-8 <= own[index] <= upper + 1e-8
            assert feasible.support(q, via="linear_certificate") >= upper - 1e-8
        # The linearized data are the derivative at the point applied to it.
        assert np.allclose(feasible.data, feasible.linearization.derivative(point))


class TestOuterApproximation:
    def test_a_polytope_from_support_values_contains_the_set(self, setting):
        model, forward, target, truth, data = setting
        problem = LinearForwardProblem(
            forward, error=Ball(forward.codomain, radius=0.2)
        )
        answer = BackusGilbertParker(problem, target, Ball(model, radius=3.0))(data)
        polytope = answer.polytope(directions(target.codomain))
        assert polytope.is_outer
        assert polytope.contains(target(truth))

    def test_more_directions_only_tighten_it(self, setting):
        model, forward, target, truth, data = setting
        answer = BackusGilbertParker(
            LinearForwardProblem(forward), target, Ball(model, radius=3.0)
        )(data)
        space = target.codomain
        few = directions(space)
        many = few + [
            space.from_components(np.array([np.cos(angle), np.sin(angle)]))
            for angle in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
        ]
        oracle = ConvexSet.from_support_function(space, answer.support_function())
        assert len(oracle.polytope(many).half_spaces) > len(
            oracle.polytope(few).half_spaces
        )

    def test_an_outer_and_an_inner_bound_cannot_be_intersected(self):
        space = EuclideanSpace(2)
        plane = HalfSpace(space, np.array([1.0, 0.0]), offset=1.0)
        outer = Polytope(space, [plane], outer=True)
        inner = Polytope(space, [plane], outer=False)
        with pytest.raises(ValueError, match="bound nothing"):
            outer & inner


class TestOneEstimator:
    """``BackusGilbertParker`` is the one public estimator: the sets decide
    the form, the form decides the algorithms, ``via=`` forces one where
    the sets allow it, and the answer carries both characterizations."""

    @pytest.fixture
    def pieces(self, setting):
        model, forward, target, truth, data = setting
        exact = LinearForwardProblem(forward)
        noisy = LinearForwardProblem(forward, error=Ball(forward.codomain, radius=0.05))
        return model, forward, target, data, exact, noisy

    def test_the_sets_choose_the_algorithms(self, pieces):
        model, forward, target, data, exact, noisy = pieces
        ball = Ball(model, radius=3.0)
        assert BackusGilbertParker(exact, target, ball).form == "closed_form"
        assert BackusGilbertParker(noisy, target, ball).form == "quadratic"
        # Exact data said explicitly, as a confidence set of radius zero.
        explicit = BackusGilbertParker(
            noisy, target, ball, noise=Ball(forward.codomain, radius=0.0)
        )
        assert explicit.form == "closed_form" and explicit.noise is None
        ellipsoid = Ellipsoid(
            model,
            LinearOperator.identity(model) * (1.0 / 9.0),
            covariance=LinearOperator.identity(model) * 9.0,
        )
        assert BackusGilbertParker(noisy, target, ellipsoid).form == "quadratic"
        # A polytope has a projection and nothing else: the splitting bounds
        # it from inside, nothing bounds it from outside, and no algorithm
        # decides membership. A Minkowski sum has the support side, which
        # also decides membership by the distance.
        box = BackusGilbertParker(noisy, target, box_prior(model))(data)
        assert box.form == "general"
        assert box.algorithms == {
            "support": ("splitting",),
            "inclusion": (),
            "feasibility": (),
            "outer_ellipsoid": (),
        }
        fat = BackusGilbertParker(
            noisy, target, Ball(model, radius=2.0) + Ball(model, radius=1.0)
        )(data)
        assert fat.algorithms == {
            "support": ("dual",),
            "inclusion": ("distance",),
            "feasibility": ("dual",),
            "outer_ellipsoid": (),
        }
        quadratic = BackusGilbertParker(noisy, target, ball)(data)
        assert quadratic.algorithms == {
            "support": ("primal", "dual", "splitting"),
            "inclusion": ("search", "distance"),
            "feasibility": ("search", "dual"),
            "outer_ellipsoid": ("s_procedure",),
        }
        closed = BackusGilbertParker(exact, target, ball)(data)
        assert closed.has_projection and isinstance(closed.ellipsoid, Ellipsoid)
        assert not quadratic.has_projection

    def test_an_algorithm_the_sets_do_not_allow_is_refused(self, pieces):
        model, forward, target, data, exact, noisy = pieces
        ball = Ball(model, radius=3.0)
        q = target.codomain.basis_vector(0)
        with pytest.raises(ValueError, match="does not apply.*primal"):
            BackusGilbertParker(noisy, target, ball)(data).support(q, via="closed_form")
        with pytest.raises(ValueError, match="does not apply"):
            BackusGilbertParker(exact, target, ball)(data).support(q, via="primal")
        with pytest.raises(ValueError, match="does not apply.*splitting"):
            BackusGilbertParker(noisy, target, box_prior(model))(data).support(
                q, via="primal"
            )
        with pytest.raises(ValueError, match="Unknown algorithm"):
            BackusGilbertParker(noisy, target, ball)(data).support(q, via="magic")
        with pytest.raises(TypeError, match="name or an instance"):
            BackusGilbertParker(noisy, target, ball)(data).support(q, via=3)
        # The dual is always allowed, balls and exact data included.
        for problem in (exact, noisy):
            feasible = BackusGilbertParker(problem, target, ball)(data)
            assert feasible.support_result(q, via="dual").algorithm == "dual"

    def test_membership_agrees_with_every_support_algorithm(self, pieces, rng):
        """Whichever algorithm computes the support, the set answers
        ``contains`` by the inclusion level, and the two sides agree."""
        model, forward, target, data, exact, noisy = pieces
        ball = Ball(model, radius=3.0)
        answer = BackusGilbertParker(noisy, target, ball)(data)
        assert answer.has_membership
        for _ in range(6):
            value = target.codomain.random(rng=rng)
            assert answer.contains(value) == answer.admits(value, via="distance")
        for via in ("primal", "dual"):
            direction = target.codomain.basis_vector(0)
            beyond = target.codomain.scale(
                1.5 * answer.support(direction, via=via) + 1.0, direction
            )
            assert not answer.contains(beyond)

    def test_general_sets_decide_membership_from_the_support_side(self, pieces, rng):
        """A Minkowski sum of balls has a support function and a maximizer
        but no level function: the dual computes the support and the
        distance decides points from it."""
        model, forward, target, data, exact, noisy = pieces
        fat = Ball(model, radius=2.0) + Ball(model, radius=1.0)
        answer = BackusGilbertParker(noisy, target, fat)(data)
        same = BackusGilbertParker(noisy, target, Ball(model, radius=3.0))(data)
        assert answer.has_membership and answer.level == 1.0
        q = target.codomain.basis_vector(0)
        assert answer.support(q) == pytest.approx(same.support(q), rel=1e-6)
        for _ in range(4):
            value = target.codomain.random(rng=rng)
            assert answer.contains(value) == same.contains(value)

    def test_the_sweep_agrees_with_asking_one_at_a_time(self, pieces):
        model, forward, target, data, exact, noisy = pieces
        estimator = BackusGilbertParker(noisy, target, Ball(model, radius=3.0))
        directions = [
            target.codomain.basis_vector(i) for i in range(target.codomain.dim)
        ]
        swept = estimator(data).support_values(directions)
        assert swept == pytest.approx([estimator(data).support(d) for d in directions])
        assert estimator(data).support_values(directions, n_jobs=2) == pytest.approx(
            swept
        )
        assert estimator(data).support_values(directions, via="dual") == pytest.approx(
            swept, rel=1e-6
        )

    def test_a_gaussian_error_is_hardened_to_the_credible_ball(self, pieces, rng):
        model, forward, target, data, exact, noisy = pieces
        error = GaussianMeasure.from_standard_deviation(forward.codomain, 0.02)
        problem = LinearForwardProblem(forward, error=error)
        estimator = BackusGilbertParker(
            problem, target, Ball(model, radius=3.0), level=0.9
        )
        assert isinstance(estimator.noise, Ball)
        assert estimator.noise.radius == pytest.approx(
            error.ambient_ball(level=0.9).radius
        )
        assert estimator.form == "quadratic"

    def test_push_forward_keeps_the_sets(self, pieces):
        model, forward, target, data, exact, noisy = pieces
        estimator = BackusGilbertParker(noisy, target, Ball(model, radius=3.0))
        further = LinearOperator.from_matrix(
            target.codomain,
            EuclideanSpace(1),
            np.ones((1, target.codomain.dim)),
            form="components",
        )
        pushed = estimator.push_forward(further)
        assert pushed.form == "quadratic"
        assert pushed.prior is estimator.prior and pushed.noise is estimator.noise
        assert pushed.property_space.dim == 1
        assert pushed.forward_problem is estimator.forward_problem

    def test_a_set_on_the_wrong_space_is_refused(self, pieces):
        model, forward, target, data, exact, noisy = pieces
        with pytest.raises(ValueError, match="model space"):
            BackusGilbertParker(noisy, target, Ball(forward.codomain, radius=1.0))
        with pytest.raises(ValueError, match="data space"):
            BackusGilbertParker(
                noisy, target, Ball(model, radius=1.0), noise=Ball(model, radius=1.0)
            )
        with pytest.raises(TypeError, match="ConvexSet"):
            BackusGilbertParker(noisy, target, "a ball")


class TestTheLevelSide:
    """Membership by the smallest fitting model, for any confidence set that
    is a sublevel set of a differentiable convex functional: the
    characterization complementary to the support function."""

    @pytest.fixture
    def pieces(self, rng):
        model = make_weighted_space()
        data_space = EuclideanSpace(3)
        property_space = EuclideanSpace(2)
        forward = LinearOperator.from_matrix(
            model, data_space, rng.normal(size=(3, model.dim)), form="galerkin"
        )
        target = LinearOperator.from_matrix(
            model, property_space, rng.normal(size=(2, model.dim)), form="galerkin"
        )
        raw = model.random(rng=rng)
        truth = model.scale(2.0 / model.norm(raw), raw)
        radius = 0.15
        noise = data_space.random(rng=rng)
        data = data_space.add(
            forward(truth),
            data_space.scale(0.6 * radius / data_space.norm(noise), noise),
        )
        return model, data_space, forward, target, truth, data, radius

    @staticmethod
    def candidates(target, truth, rng, count=6):
        space = target.codomain
        center = target(truth)
        return [center] + [
            space.axpy(0.5, space.random(rng=rng), space.copy(center))
            for _ in range(count - 1)
        ]

    def test_the_search_and_the_distance_agree_on_a_ball(self, pieces, rng):
        """The same membership two ways: the one-multiplier search in the
        model space, and the projection dual on the support function."""
        model, data_space, forward, target, truth, data, radius = pieces
        problem = LinearForwardProblem(forward, error=Ball(data_space, radius=radius))
        feasible = BackusGilbertParker(problem, target, Ball(model, radius=3.0))(data)
        assert feasible.algorithms["inclusion"] == ("search", "distance")
        for value in self.candidates(target, truth, rng):
            assert feasible.admits(value, via="search") == feasible.admits(
                value, via="distance"
            )
        assert not feasible.is_empty()
        assert feasible.admits(target(truth))

    def test_a_sublevel_set_equal_to_the_ball_gives_the_same_answer(self, pieces, rng):
        """``||v||^4 <= r^4`` is the ball; given with a gradient only, so the
        general tier is what is tested: the dual on a converted support
        function, the search through L-BFGS."""
        model, data_space, forward, target, truth, data, radius = pieces
        same_ball = quartic_ball(data_space, radius)
        problem = LinearForwardProblem(forward, error=same_ball)
        prior = Ball(model, radius=3.0)
        estimator = BackusGilbertParker(problem, target, prior)
        assert estimator.form == "general"
        feasible = estimator(data)
        assert feasible.algorithms["support"][0] == "dual"
        assert feasible.algorithms["inclusion"][0] == "search"
        assert feasible.has_support_function and feasible.has_level_function
        assert feasible.has_maximizer
        assert not feasible.is_empty()
        assert feasible.contains(target(truth))
        as_ball = BackusGilbertParker(
            LinearForwardProblem(forward, error=Ball(data_space, radius=radius)),
            target,
            prior,
        )(data)
        q = target.codomain.basis_vector(0)
        assert feasible.support(q) == pytest.approx(as_ball.support(q), rel=1e-4)
        for value in self.candidates(target, truth, rng, count=4):
            expected = as_ball.inclusion_level(value)
            if np.isfinite(expected):
                assert feasible.inclusion_level(value) == pytest.approx(
                    expected, rel=1e-4
                )
            else:
                assert feasible.inclusion_level(value) == float("inf")

    def test_a_gaussian_error_taken_as_its_ellipsoid(self, pieces, rng):
        """The credible ellipsoid sits inside the ambient ball, so it admits
        no more than the ball does and its inclusion levels are no smaller;
        and as the error vanishes both approach the closed form."""
        model, data_space, forward, target, truth, data, radius = pieces
        prior = Ball(model, radius=3.0)
        exact = BackusGilbertParker(LinearForwardProblem(forward), target, prior)
        previous = None
        for sigma in (0.05, 0.005):
            error = GaussianMeasure.from_standard_deviation(data_space, sigma)
            problem = LinearForwardProblem(forward, error=error)
            with_ball = BackusGilbertParker(problem, target, prior, level=0.9)
            ellipsoid = error.credible_set(level=0.9)
            with_ellipsoid = BackusGilbertParker(
                problem, target, prior, noise=ellipsoid
            )
            assert with_ellipsoid.form == "quadratic"
            for value in self.candidates(target, truth, rng, count=4):
                ball_level = with_ball(data).inclusion_level(value)
                ellipsoid_level = with_ellipsoid(data).inclusion_level(value)
                assert ellipsoid_level >= ball_level * (1.0 - 1e-5)
            # With more data than the kernel of T has dimensions the exact
            # set is flat, and the limit is checked at points of it: its
            # centre and two of its extreme points.
            gap = 0.0
            for value in [exact(data).ellipsoid.center] + [
                target(exact(data).extremal_model(target.codomain.basis_vector(i)))
                for i in range(2)
            ]:
                limit = exact(data).inclusion_level(value)
                assert np.isfinite(limit)
                gap = max(gap, abs(with_ellipsoid(data).inclusion_level(value) - limit))
            assert gap > 0.0
            if previous is not None:
                assert gap < previous
            previous = gap

    def test_the_answer_is_a_sublevel_set(self, pieces, rng):
        model, data_space, forward, target, truth, data, radius = pieces
        problem = LinearForwardProblem(forward, error=Ball(data_space, radius=radius))
        estimator = BackusGilbertParker(problem, target, Ball(model, radius=3.0))
        feasible = estimator(data)
        # The general contract: the answer is the sublevel set of the prior's
        # level function at the fitting model, at the prior's own level --
        # a ball's squared radius.
        assert feasible.has_level_function and feasible.level == 9.0
        as_set = SublevelSet(feasible.level_function(), level=feasible.level)
        functional = feasible.level_function()
        for value in self.candidates(target, truth, rng):
            assert functional(value) == pytest.approx(feasible.inclusion_level(value))
            assert as_set.contains(value) == feasible.admits(value)
            assert feasible.contains(value) == feasible.admits(value)

    def test_the_extent_is_the_support_interval_on_a_scalar_property(self, pieces, rng):
        """One-dimensional property: the two ends of the line are the two
        support values, so the inner and outer bounds meet."""
        model, data_space, forward, target, truth, data, radius = pieces
        scalar = LinearOperator.from_matrix(
            model, EuclideanSpace(1), rng.normal(size=(1, model.dim)), form="galerkin"
        )
        problem = LinearForwardProblem(forward, error=Ball(data_space, radius=radius))
        feasible = BackusGilbertParker(problem, scalar, Ball(model, radius=3.0))(data)
        q = EuclideanSpace(1).basis_vector(0)
        lower, upper = feasible.extent(q)
        assert lower < upper
        assert upper == pytest.approx(feasible.support(q), rel=1e-5)
        assert lower == pytest.approx(-feasible.support(-q), rel=1e-5)

    def test_the_extent_is_an_inner_bound_in_more_dimensions(self, pieces):
        model, data_space, forward, target, truth, data, radius = pieces
        problem = LinearForwardProblem(forward, error=Ball(data_space, radius=radius))
        feasible = BackusGilbertParker(problem, target, Ball(model, radius=3.0))(data)
        for index in range(2):
            q = target.codomain.basis_vector(index)
            lower, upper = feasible.extent(q)
            assert lower < upper
            assert upper <= feasible.support(q) * (1.0 + 1e-6) + 1e-9
            assert lower >= -feasible.support(-q) * (1.0 + 1e-6) - 1e-9

    def test_a_polytope_prior_is_bounded_from_inside_only(self, pieces):
        model, data_space, forward, target, truth, data, radius = pieces
        problem = LinearForwardProblem(forward, error=Ball(data_space, radius=radius))
        feasible = BackusGilbertParker(problem, target, box_prior(model))(data)
        assert feasible.has_support_function and not feasible.has_membership
        q = target.codomain.basis_vector(0)
        inner, outer = feasible.support_bracket(q)
        assert np.isfinite(inner) and outer == np.inf
        with pytest.raises(NotImplementedError, match="No algorithm answers"):
            feasible.admits(target.codomain.zero())
        with pytest.raises(NotImplementedError, match="No algorithm answers"):
            feasible.extent(q)


class TestTheResultIsWhatGetsProbed:
    """``BackusGilbertParker(...)(data)`` is a ``FeasiblePropertySet``: a
    convex set that declares what it can do and answers every question
    itself, as a posterior measure does on the Bayesian side."""

    @pytest.fixture
    def pieces(self, setting):
        model, forward, target, truth, data = setting
        exact = LinearForwardProblem(forward)
        noisy = LinearForwardProblem(forward, error=Ball(forward.codomain, radius=0.05))
        return model, forward, target, truth, data, exact, noisy

    @staticmethod
    def flags(feasible):
        return (
            feasible.has_membership,
            feasible.has_projection,
            feasible.has_support_function,
            feasible.has_maximizer,
            feasible.has_level_function,
        )

    def test_each_form_declares_its_characterizations(self, pieces):
        model, forward, target, truth, data, exact, noisy = pieces
        ball = Ball(model, radius=3.0)
        closed = BackusGilbertParker(exact, target, ball)(data)
        assert isinstance(closed, FeasiblePropertySet)
        assert self.flags(closed) == (True, True, True, True, True)
        quadratic = BackusGilbertParker(noisy, target, ball)(data)
        assert self.flags(quadratic) == (True, False, True, True, True)
        fat = Ball(model, radius=2.0) + Ball(model, radius=1.0)
        general = BackusGilbertParker(noisy, target, fat)(data)
        assert self.flags(general) == (True, False, True, True, True)
        box = BackusGilbertParker(noisy, target, box_prior(model))(data)
        assert self.flags(box) == (False, False, True, True, False)
        for refused in (
            lambda: box.contains(target(truth)),
            lambda: box.extent(target.codomain.basis_vector(0)),
            lambda: box.level_function(),
            lambda: quadratic.project(target(truth)),
            lambda: quadratic.ellipsoid,
        ):
            with pytest.raises(NotImplementedError):
                refused()

    def test_the_closed_forms_extremal_model_attains_the_support(self, pieces, rng):
        model, forward, target, truth, data, exact, noisy = pieces
        feasible = BackusGilbertParker(exact, target, Ball(model, radius=3.0))(data)
        for _ in range(4):
            q = target.codomain.random(rng=rng)
            extremal = feasible.extremal_model(q)
            assert model.norm(extremal) <= 3.0 * (1.0 + 1e-9)
            assert forward.codomain.norm(
                forward.codomain.subtract(forward(extremal), data)
            ) < 1e-8 * forward.codomain.norm(data)
            assert target.codomain.inner_product(q, target(extremal)) == pytest.approx(
                feasible.support(q), rel=1e-8
            )
            assert (
                target.codomain.norm(
                    target.codomain.subtract(
                        feasible.support_maximizer(q), target(extremal)
                    )
                )
                < 1e-10
            )
            # The support function object's subgradient is that maximizer.
            assert (
                target.codomain.norm(
                    target.codomain.subtract(
                        feasible.support_function().subgradient(q), target(extremal)
                    )
                )
                < 1e-10
            )

    def test_the_result_pushes_forward(self, pieces):
        model, forward, target, truth, data, exact, noisy = pieces
        feasible = BackusGilbertParker(noisy, target, Ball(model, radius=3.0))(data)
        summary = LinearOperator.from_matrix(
            target.codomain,
            EuclideanSpace(1),
            np.ones((1, target.codomain.dim)),
            form="components",
        )
        pushed = feasible.push_forward(summary)
        assert pushed.domain.dim == 1
        assert pushed.form == feasible.form
        q = EuclideanSpace(1).basis_vector(0)
        # h_{T S}(q) == h_S(T* q), exactly.
        assert pushed.support(q) == pytest.approx(
            feasible.support(summary.adjoint(q)), rel=1e-6
        )

    def test_an_exclusion_certificate_and_the_two_bounds(self, pieces, rng):
        model, forward, target, truth, data, exact, noisy = pieces
        feasible = BackusGilbertParker(noisy, target, Ball(model, radius=3.0))(data)
        space = target.codomain
        far = space.scale(50.0, space.random(rng=rng))
        assert feasible.outside(far, directions(space))
        assert not feasible.outside(target(truth), directions(space))
        outer = feasible.polytope(directions(space))
        assert outer.is_outer and outer.contains(target(truth))
        assert feasible.contains(target(truth))
        angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
        probes = [
            space.from_components(np.array([np.cos(a), np.sin(a)])) for a in angles
        ]
        inner = feasible.inner_hull(probes)
        assert not inner.is_outer
        for direction in probes:
            contact = feasible.support_maximizer(direction)
            assert inner.contains(contact, rtol=1e-8) and outer.contains(contact)
            assert feasible.admits(contact, rtol=1e-6)

    def test_emptiness_is_answered_by_every_algorithm(self, pieces):
        model, forward, target, truth, data, exact, noisy = pieces
        tight, roomy = Ball(model, radius=1e-3), Ball(model, radius=100.0)
        assert BackusGilbertParker(exact, target, tight)(data).is_empty()
        assert not BackusGilbertParker(exact, target, roomy)(data).is_empty()
        for via in ("search", "dual"):
            assert BackusGilbertParker(noisy, target, tight)(data).is_empty(via=via)
            assert not BackusGilbertParker(noisy, target, roomy)(data).is_empty(via=via)


class TestQuadraticSets:
    """The primal on ellipsoids: written in the sets' own inner products, so
    an ellipsoidal prior or confidence set takes the quadratic tier,
    agreeing with the dual, which computes the same set another way, and
    with the ball where the two coincide."""

    @pytest.fixture
    def pieces(self, setting, rng):
        model, forward, target, truth, data = setting
        error = GaussianMeasure.from_standard_deviations(
            forward.codomain, np.linspace(0.02, 0.08, forward.codomain.dim)
        )
        problem = LinearForwardProblem(forward, error=error)
        credible = error.credible_set(level=0.9)
        return model, forward, target, truth, data, problem, credible

    @staticmethod
    def agree(first, second, directions, rel, *, via=None):
        for direction in directions:
            assert first.support(direction) == pytest.approx(
                second.support(direction, via=via), rel=rel
            )

    def test_an_ellipsoid_written_as_a_ball_is_the_ball(self, setting, rng):
        model, forward, target, truth, data = setting
        data_space = forward.codomain
        radius = 0.1
        problem = LinearForwardProblem(forward, error=Ball(data_space, radius=radius))
        as_ball = BackusGilbertParker(problem, target, Ball(model, radius=3.0))(data)
        precision = (
            LinearOperator.identity(data_space) * (1.0 / radius**2)
        ).with_traits(Traits.POSITIVE_DEFINITE)
        covariance = (LinearOperator.identity(data_space) * radius**2).with_traits(
            Traits.POSITIVE_DEFINITE
        )
        as_ellipsoid = BackusGilbertParker(
            problem,
            target,
            Ball(model, radius=3.0),
            noise=Ellipsoid(data_space, precision, covariance=covariance),
        )(data)
        assert as_ellipsoid.form == "quadratic"
        self.agree(as_ellipsoid, as_ball, directions(target.codomain), 1e-8)
        for _ in range(4):
            value = target.codomain.random(rng=rng)
            assert as_ellipsoid.inclusion_level(value) == pytest.approx(
                as_ball.inclusion_level(value), rel=1e-8
            )
        assert as_ellipsoid.is_empty() == as_ball.is_empty()

    def test_an_ellipsoidal_confidence_set_agrees_with_the_dual(self, pieces):
        model, forward, target, truth, data, problem, credible = pieces
        prior = Ball(model, radius=3.0)
        primal = BackusGilbertParker(problem, target, prior, noise=credible)(data)
        assert primal.form == "quadratic"
        self.agree(primal, primal, directions(target.codomain), 1e-5, via="dual")
        # The extremal model attains the bound and lies in both sets.
        q = target.codomain.basis_vector(0)
        extremal = primal.extremal_model(q)
        assert model.norm(extremal) <= 3.0 * (1.0 + 1e-8)
        assert credible.contains(
            forward.codomain.subtract(data, forward(extremal)), rtol=1e-6
        )
        assert target.codomain.inner_product(q, target(extremal)) == pytest.approx(
            primal.support(q), rel=1e-4
        )
        assert primal.contains(target(truth))
        assert not primal.is_empty()

    def test_an_ellipsoidal_prior_agrees_with_the_dual(self, setting):
        model, forward, target, truth, data = setting
        prior = anisotropic_ellipsoid(model, [36.0, 16.0, 16.0, 9.0])
        problem = LinearForwardProblem(
            forward, error=Ball(forward.codomain, radius=0.1)
        )
        primal = BackusGilbertParker(problem, target, prior)(data)
        assert primal.form == "quadratic" and primal.level == 1.0
        self.agree(primal, primal, directions(target.codomain), 1e-5, via="dual")
        assert primal.contains(target(truth))
        q = target.codomain.basis_vector(1)
        extremal = primal.extremal_model(q)
        assert prior.contains(extremal, rtol=1e-6)
        assert forward.codomain.norm(
            forward.codomain.subtract(data, forward(extremal))
        ) <= 0.1 * (1.0 + 1e-6)

    def test_ellipsoids_on_both_sides_and_off_center_sets(self, pieces, rng):
        model, forward, target, truth, data, problem, credible = pieces
        center = model.scale(0.3, truth)
        prior = anisotropic_ellipsoid(
            model, np.linspace(9.0, 36.0, model.dim), center=center
        )
        primal = BackusGilbertParker(problem, target, prior, noise=credible)(data)
        self.agree(primal, primal, directions(target.codomain), 1e-5, via="dual")
        assert primal.has_membership
        for _ in range(4):
            value = target.codomain.random(rng=rng)
            inside = primal.contains(value)
            assert (
                inside == (not primal.outside(value, directions(target.codomain)))
                or inside is False
            )
        # A ball prior off the origin, against the dual.
        shifted = Ball(model, radius=2.0, center=center)
        primal = BackusGilbertParker(problem, target, shifted)(data)
        self.agree(primal, primal, directions(target.codomain), 1e-5, via="dual")


class TestGeneralSets:
    """The general tier: a prior that is not a ball, the contract for the
    answer's level function, and sets known through one description."""

    @pytest.fixture
    def pieces(self, rng):
        model = make_weighted_space()
        data_space = EuclideanSpace(3)
        property_space = EuclideanSpace(2)
        forward = LinearOperator.from_matrix(
            model, data_space, rng.normal(size=(3, model.dim)), form="galerkin"
        )
        target = LinearOperator.from_matrix(
            model, property_space, rng.normal(size=(2, model.dim)), form="galerkin"
        )
        raw = model.random(rng=rng)
        truth = model.scale(2.0 / model.norm(raw), raw)
        radius = 0.15
        noise = data_space.random(rng=rng)
        data = data_space.add(
            forward(truth),
            data_space.scale(0.6 * radius / data_space.norm(noise), noise),
        )
        problem = LinearForwardProblem(forward, error=Ball(data_space, radius=radius))
        return model, data_space, forward, target, truth, data, problem

    @staticmethod
    def candidates(target, truth, rng, count=5):
        space = target.codomain
        center = target(truth)
        return [center] + [
            space.axpy(0.5, space.random(rng=rng), space.copy(center))
            for _ in range(count - 1)
        ]

    def test_the_general_contract_on_a_ball_prior(self, pieces, rng):
        """The answer's level function is the prior's at the fitting model,
        against the prior's level: for a ball, the squared norm against the
        squared radius."""
        model, data_space, forward, target, truth, data, problem = pieces
        feasible = BackusGilbertParker(problem, target, Ball(model, radius=3.0))(data)
        assert feasible.level == 9.0
        for value in self.candidates(target, truth, rng):
            level = feasible.inclusion_level(value)
            if np.isfinite(level):
                assert feasible.level_function()(value) == pytest.approx(level)
            assert feasible.contains(value) == (level <= 9.0 * (1.0 + 1e-8))

    def test_a_quartic_prior_equal_to_the_ball_gives_the_balls_answers(
        self, pieces, rng
    ):
        """``||m||^4 <= r^4`` is the ball, given as a sublevel set with a
        gradient only: the dual on a converted support function and the
        search through L-BFGS, against the quadratic tier on the ball."""
        model, data_space, forward, target, truth, data, problem = pieces
        estimator = BackusGilbertParker(problem, target, quartic_ball(model, 3.0))
        assert estimator.form == "general"
        feasible = estimator(data)
        assert feasible.algorithms["support"][0] == "dual"
        assert feasible.algorithms["inclusion"][0] == "search"
        assert feasible.level == 3.0**4
        reference = BackusGilbertParker(problem, target, Ball(model, radius=3.0))(data)
        for value in self.candidates(target, truth, rng):
            expected = reference.inclusion_level(value)
            got = feasible.inclusion_level(value)
            if np.isfinite(expected):
                assert got == pytest.approx(expected**2, rel=1e-4)
            else:
                assert got == float("inf")
            assert feasible.contains(value) == reference.contains(value)
        assert not feasible.is_empty()
        for q in directions(target.codomain):
            assert feasible.support(q) == pytest.approx(reference.support(q), rel=1e-4)
            extremal = feasible.extremal_model(q)
            assert model.norm(extremal) <= 3.0 * (1.0 + 1e-4)
            assert forward.codomain.norm(
                forward.codomain.subtract(data, forward(extremal))
            ) <= 0.15 * (1.0 + 1e-4)
        q = target.codomain.basis_vector(0)
        lower, upper = feasible.extent(q)
        assert lower < upper
        assert upper <= reference.support(q) * (1.0 + 1e-5) + 1e-9
        assert lower >= -reference.support(-q) * (1.0 + 1e-5) - 1e-9
        outer = feasible.polytope(directions(target.codomain))
        assert outer.is_outer and outer.contains(target(truth))

    def test_a_quartic_confidence_set_takes_the_dual(self, pieces):
        model, data_space, forward, target, truth, data, problem = pieces
        noise = quartic_ball(forward.codomain, 0.15)
        estimator = BackusGilbertParker(
            problem, target, Ball(model, radius=3.0), noise=noise
        )
        assert estimator.form == "general"
        feasible = estimator(data)
        reference = BackusGilbertParker(problem, target, Ball(model, radius=3.0))(data)
        for direction in directions(target.codomain)[:3]:
            assert feasible.support(direction) == pytest.approx(
                reference.support(direction), rel=1e-4
            )

    def test_an_ellipsoidal_prior_through_both_membership_algorithms(self, pieces, rng):
        model, data_space, forward, target, truth, data, problem = pieces
        prior = anisotropic_ellipsoid(model, np.linspace(9.0, 36.0, model.dim))
        feasible = BackusGilbertParker(problem, target, prior)(data)
        assert feasible.level == 1.0
        for value in self.candidates(target, truth, rng):
            assert feasible.admits(value, via="search") == feasible.admits(
                value, via="distance"
            )
        assert feasible.contains(target(truth))

    def test_slack_data_leave_the_priors_own_support_point(self, pieces):
        """A confidence set wide enough that the data never bite: the
        answer is the prior's support point, through the converted support
        function."""
        model, data_space, forward, target, truth, data, problem = pieces
        prior = quartic_ball(model, 0.05)
        noise = Ball(forward.codomain, radius=1e3)
        feasible = BackusGilbertParker(problem, target, prior, noise=noise)(data)
        direction = target.codomain.basis_vector(0)
        pulled = target.adjoint(direction)
        assert feasible.support(direction) == pytest.approx(
            0.05 * model.norm(pulled), rel=1e-5
        )
        assert model.norm(feasible.extremal_model(direction)) == pytest.approx(
            0.05, rel=1e-5
        )


class TestBundleMethod:
    """The minimizer the dual is built on."""

    def test_it_minimizes_a_nonsmooth_convex_function(self):
        """``|x - a|_1 + |x|^2/2``, whose minimizer is ``clip(a, -1, 1)``.

        Not soft-thresholding, which is the answer to a different problem and
        was the first reference tried here.
        """
        from pygeoinf2.numerics.convex import ProximalBundleMethod

        space = EuclideanSpace(4)
        anchor = np.random.default_rng(0).normal(size=4)
        functional = Functional.from_callables(
            space,
            lambda x: float(np.abs(x - anchor).sum() + 0.5 * x @ x),
            gradient=lambda x: np.sign(x - anchor) + x,
        )
        result = ProximalBundleMethod(tolerance=1e-8, max_iterations=400).minimize(
            functional, space.zero()
        )
        best = np.clip(anchor, -1.0, 1.0)
        assert result.converged
        assert result.value == pytest.approx(
            float(np.abs(best - anchor).sum() + 0.5 * best @ best), abs=1e-6
        )
        assert np.allclose(result.minimizer, best, atol=1e-5)

    def test_the_gap_certifies_the_answer(self):
        from pygeoinf2.numerics.convex import ProximalBundleMethod

        space = EuclideanSpace(4)
        rng = np.random.default_rng(1)
        root = rng.normal(size=(4, 4))
        matrix = root @ root.T + 4.0 * np.identity(4)
        offset = rng.normal(size=4)
        functional = Functional.from_callables(
            space,
            lambda x: float(0.5 * x @ matrix @ x - offset @ x),
            gradient=lambda x: matrix @ x - offset,
        )
        result = ProximalBundleMethod(tolerance=1e-12, max_iterations=600).minimize(
            functional, space.zero()
        )
        best = np.linalg.solve(matrix, offset)
        assert result.value == pytest.approx(
            float(0.5 * best @ matrix @ best - offset @ best), abs=1e-7
        )
        assert result.gap >= 0.0

    def test_a_nonsense_descent_fraction_is_refused(self):
        from pygeoinf2.numerics.convex import ProximalBundleMethod

        with pytest.raises(ValueError, match="descent fraction"):
            ProximalBundleMethod(descent=1.5)


class TestDual:
    def test_it_agrees_with_the_primal(self, setting):
        """Two algorithms with nothing in common: a one-multiplier
        minimization in the model space against a nonsmooth convex
        minimization in the data space."""
        model, forward, target, truth, data = setting
        prior = Ball(model, radius=3.0)
        for radius in (0.3, 0.05):
            problem = LinearForwardProblem(
                forward, error=Ball(forward.codomain, radius=radius)
            )
            feasible = BackusGilbertParker(problem, target, prior)(data)
            for direction in directions(target.codomain):
                assert feasible.support(direction, via="dual") == pytest.approx(
                    feasible.support(direction, via="primal"), rel=1e-6
                )

    def test_a_ball_written_as_an_ellipsoid_gives_the_same_answer(self, setting):
        model, forward, target, truth, data = setting
        radius = 3.0
        covariance = LinearOperator.self_adjoint(
            model,
            lambda v: model.scale(radius**2, v),
            traits=Traits.POSITIVE_DEFINITE,
        )
        precision = LinearOperator.self_adjoint(
            model,
            lambda v: model.scale(1.0 / radius**2, v),
            traits=Traits.POSITIVE_DEFINITE,
        )
        problem = LinearForwardProblem(
            forward, error=Ball(forward.codomain, radius=0.1)
        )
        as_ball = BackusGilbertParker(problem, target, Ball(model, radius=radius))(data)
        as_ellipsoid = BackusGilbertParker(
            problem, target, Ellipsoid(model, precision, covariance=covariance)
        )(data)
        for direction in directions(target.codomain):
            assert as_ellipsoid.support(direction, via="dual") == pytest.approx(
                as_ball.support(direction, via="dual"), rel=1e-6
            )

    def test_an_empty_feasible_set_is_reported(self, setting):
        """An unbounded dual is a statement about the problem, and a large
        negative number is a perfectly plausible-looking support value."""
        model, forward, target, truth, data = setting
        problem = LinearForwardProblem(
            forward, error=Ball(forward.codomain, radius=1e-6)
        )
        feasible = BackusGilbertParker(problem, target, Ball(model, radius=0.01))(data)
        with pytest.raises(ValueError, match="empty"):
            feasible.support(target.codomain.basis_vector(0), via="dual")
        assert feasible.is_empty(via="dual")

    def test_the_certificate_is_a_weighting_of_the_data(self, setting):
        model, forward, target, truth, data = setting
        problem = LinearForwardProblem(
            forward, error=Ball(forward.codomain, radius=0.1)
        )
        feasible = BackusGilbertParker(problem, target, Ball(model, radius=3.0))(data)
        direction = target.codomain.basis_vector(0)
        certificate = feasible.certificate(direction, via="dual")
        # any certificate gives a valid bound; this one is the best
        cost = Dual().cost(feasible, direction)
        assert cost(certificate) <= cost(forward.codomain.zero()) + 1e-9
        assert cost(certificate) == pytest.approx(
            feasible.support(direction, via="dual")
        )


class TestInclusionWithErrors:
    """Set inclusion as a constrained optimization, with noisy data.

    The complement of the support-function machinery: a support function
    bounds the set from outside one direction at a time, and this decides
    membership exactly one point at a time. Only the second can produce an
    inner bound from candidates, and the two must agree about every point
    they both have an opinion on.
    """

    @pytest.fixture
    def noisy(self, rng):
        model = make_weighted_space()
        data_space = EuclideanSpace(3)
        property_space = EuclideanSpace(2)
        forward = LinearOperator.from_matrix(
            model, data_space, rng.normal(size=(3, model.dim)), form="galerkin"
        )
        target = LinearOperator.from_matrix(
            model, property_space, rng.normal(size=(2, model.dim)), form="galerkin"
        )
        raw = model.random(rng=rng)
        truth = model.scale(2.0 / model.norm(raw), raw)
        radius = 0.15
        noise = data_space.random(rng=rng)
        data = data_space.add(
            forward(truth),
            data_space.scale(0.6 * radius / data_space.norm(noise), noise),
        )
        problem = LinearForwardProblem(forward, error=Ball(data_space, radius=radius))
        estimator = BackusGilbertParker(problem, target, Ball(model, radius=3.0))
        return model, forward, target, truth, data, estimator

    def test_the_truth_is_admitted(self, noisy):
        model, forward, target, truth, data, estimator = noisy
        assert estimator(data).admits(target(truth))
        assert estimator(data).inclusion_level(target(truth)) <= (
            model.squared_norm(truth) + 1e-6
        )

    def test_it_reduces_to_the_error_free_test(self, noisy, rng):
        """As the noise ball shrinks, with the difference tracking its radius:
        the exact-data test is the noisy one with the confidence set
        collapsed to a point."""
        model, forward, target, truth, data, _ = noisy
        prior = Ball(model, radius=3.0)
        exact = BackusGilbertParker(LinearForwardProblem(forward), target, prior)
        space = target.codomain
        for radius in (1e-2, 1e-4):
            noisy_estimator = BackusGilbertParker(
                LinearForwardProblem(
                    forward, error=Ball(forward.codomain, radius=radius)
                ),
                target,
                prior,
            )
            values = [target(truth)] + [
                space.from_components(
                    space.to_components(target(truth)) + rng.normal(size=space.dim)
                )
                for _ in range(8)
            ]
            for value in values:
                limit = exact(data).inclusion_level(value)
                noisy_level = noisy_estimator(data).inclusion_level(value)
                if np.isfinite(limit):
                    assert np.sqrt(noisy_level) == pytest.approx(
                        np.sqrt(limit), rel=10.0 * radius
                    )
                else:
                    # Off the flat exact set: unreachable once the noise ball
                    # is small enough, which it is here.
                    assert noisy_level == float("inf")

    def test_it_never_admits_what_the_support_function_excludes(self, noisy, rng):
        """The two descriptions of one set, checked against each other.

        A minimum-level computation against a directional bound. Neither
        can be adjusted to agree with the other, so an inconsistency would
        name which is wrong.
        """
        model, forward, target, truth, data, estimator = noisy
        space = target.codomain
        answer = estimator(data)
        angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
        probes = [
            space.from_components(np.array([np.cos(angle), np.sin(angle)]))
            for angle in angles
        ]
        outer = answer.polytope(probes)

        admitted = excluded = 0
        for _ in range(120):
            value = space.from_components(
                space.to_components(target(truth)) + rng.normal(size=2) * 2.0
            )
            if answer.admits(value):
                admitted += 1
                assert outer.contains(value)
            if answer.outside(value, probes):
                excluded += 1
                assert not answer.admits(value)
        assert admitted > 0
        assert excluded > 0

    def test_an_unreachable_value_is_proved_unreachable(self, noisy):
        """Infinity is a proof, not a failure to converge: whatever of the
        residual lies outside the range of ``A P`` cannot be fitted however
        large the model is allowed to be."""
        model, forward, target, truth, data, estimator = noisy
        space = target.codomain
        far = space.from_components(
            space.to_components(target(truth)) + np.array([1e4, 0.0])
        )
        assert estimator(data).inclusion_level(far) == float("inf")
        assert not estimator(data).admits(far)

    def test_the_inner_hull_sits_inside_the_outer_bound(self, noisy, rng):
        """The inner hull inside, the certified outer polytope outside: the
        set sandwiched between the two."""
        model, forward, target, truth, data, estimator = noisy
        space = target.codomain
        candidates = [
            space.from_components(
                space.to_components(target(truth)) + rng.normal(size=2) * 2.0
            )
            for _ in range(400)
        ]
        hull = estimator(data).inner_hull(candidates, source="admitted")
        assert not hull.is_outer
        assert hull.contains(target(truth))

        answer = estimator(data)
        angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
        probes = [
            space.from_components(np.array([np.cos(angle), np.sin(angle)]))
            for angle in angles
        ]
        outer = answer.polytope(probes)
        for candidate in candidates:
            if hull.contains(candidate):
                assert outer.contains(candidate)

    def test_too_few_admissible_candidates_is_refused(self, noisy):
        model, forward, target, truth, data, estimator = noisy
        space = target.codomain
        far = [space.from_components(np.array([1e4, 1e4])) for _ in range(10)]
        with pytest.raises(ValueError, match="not enough"):
            estimator(data).inner_hull(far, source="admitted")


class TestHardeningTheError:
    """The bridge from a Gaussian error to the ball the certificate needs."""

    def test_no_error_measure_gives_the_degenerate_ball(self, setting):
        model, forward, _, _, data = setting
        ball = harden_error(LinearForwardProblem(forward), level=0.95)
        assert ball.radius == 0.0
        assert ball.contains(forward.codomain.zero())

    @pytest.mark.parametrize("build", [lambda: EuclideanSpace(4), make_weighted_space])
    def test_the_ball_carries_the_probability_it_claims(self, build, rng):
        """An anisotropic error on a weighted space, where a rule reading
        the component matrix's diagonal as the variance would be wrong."""
        data_space = build()
        galerkin = np.diag([0.5, 2.0, 0.1, 3.0])
        components = np.column_stack(
            [data_space.solve_gram(column) for column in galerkin.T]
        )
        error = GaussianMeasure.from_covariance_matrix(
            data_space, components, form="components"
        )
        forward = LinearOperator.from_matrix(
            EuclideanSpace(2), data_space, np.eye(4, 2), form="components"
        )
        problem = LinearForwardProblem(forward, error=error)

        level = 0.9
        ball = harden_error(problem, level=level)
        draws = error.samples(20000, rng=rng)
        covered = np.mean([ball.contains(draw) for draw in draws])
        assert covered == pytest.approx(level, abs=0.02)


class TestSupportValuesSweep:
    """Neighboring directions have neighboring certificates, so each dual
    minimization started from the last one's answer is a correction rather
    than a fresh problem."""

    @pytest.fixture
    def dual(self, rng):
        model = EuclideanSpace(12)
        data_space = EuclideanSpace(5)
        property_space = EuclideanSpace(2)
        forward = LinearOperator.from_matrix(
            model, data_space, rng.standard_normal((5, 12)), form="components"
        )
        target = LinearOperator.from_matrix(
            model, property_space, rng.standard_normal((2, 12)), form="components"
        )
        data = forward(model.random(rng=rng))
        problem = LinearForwardProblem(forward, error=Ball(data_space, radius=0.05))
        return (
            BackusGilbertParker(problem, target, Ball(model, radius=5.0)),
            property_space,
            data,
            data_space,
        )

    @staticmethod
    def directions(space, count):
        angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
        return [
            space.from_components(np.array([np.cos(angle), np.sin(angle)]))
            for angle in angles
        ]

    def test_the_warm_start_does_not_change_the_answers(self, dual):
        """Which is the only thing that would make the saving worthless."""
        estimator, space, data, data_space = dual
        directions = self.directions(space, 8)
        warm = estimator(data).support_values(directions, via="dual")
        cold = estimator(data).support_values(
            directions, via=Dual(start=data_space.zero())
        )
        assert warm == pytest.approx(cold, abs=1e-5)

    def test_it_agrees_with_asking_one_at_a_time(self, dual):
        estimator, space, data, _ = dual
        directions = self.directions(space, 6)
        swept = estimator(data).support_values(directions, via="dual")
        singly = np.array(
            [estimator(data).support(direction, via="dual") for direction in directions]
        )
        assert swept == pytest.approx(singly, abs=1e-5)

    def test_no_directions_is_no_values(self, dual):
        estimator, _, data, _ = dual
        assert estimator(data).support_values([], via="dual").size == 0

    def test_an_empty_feasible_set_is_still_reported(self, dual, rng):
        estimator, space, data, _ = dual
        tight = BackusGilbertParker(
            estimator.forward_problem,
            estimator.property_operator,
            Ball(estimator.forward_problem.model_space, radius=1e-4),
        )
        with pytest.raises(ValueError, match="empty"):
            tight(data).support_values(self.directions(space, 3), via="dual")


class TestTheDualCostMemo:
    """The dual cost fuses value and subgradient behind a one-entry memo.

    Keyed on identity, which is only meaningful among *live* objects: the
    memo keeps the certificate itself, so the reference that makes the
    identity test meaningful is the same reference that keeps the address
    from being recycled.
    """

    @pytest.fixture
    def oracle(self, rng):
        model = EuclideanSpace(8)
        data_space = EuclideanSpace(3)
        property_space = EuclideanSpace(1)
        forward = LinearOperator.from_matrix(
            model, data_space, rng.standard_normal((3, 8)), form="components"
        )
        target = LinearOperator.from_matrix(
            model, property_space, rng.standard_normal((1, 8)), form="components"
        )
        problem = LinearForwardProblem(forward, error=Ball(data_space, radius=0.1))
        data = forward(model.random(rng=rng))
        feasible = BackusGilbertParker(problem, target, Ball(model, radius=1.0))(data)
        return feasible, property_space.basis_vector(0), data

    def test_the_cache_keeps_the_certificate_alive(self, oracle):
        import gc
        import weakref

        feasible, direction, data = oracle
        cost = Dual().cost(feasible, direction)
        certificate = data.copy()
        cost(certificate)
        watch = weakref.ref(certificate)
        del certificate
        gc.collect()
        assert watch() is not None

    def test_a_recycled_address_gives_no_stale_gradient(self, oracle, rng):
        """The reproduction: a fresh cost is the reference at every point."""
        import gc

        feasible, direction, data = oracle
        cached = Dual().cost(feasible, direction)
        for _ in range(10):
            certificate = rng.standard_normal(3)
            fresh = Dual().cost(feasible, direction)
            assert np.allclose(
                cached.gradient(certificate), fresh.gradient(certificate)
            )
            assert cached(certificate) == pytest.approx(fresh(certificate))
            del certificate
            gc.collect()
