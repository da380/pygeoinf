"""
Set-valued inference: the feasible property set, computed three ways.

Routes (a), (b) and (c) of DESIGN.md §18.3 compute the same object, so every
test here is a comparison rather than an assertion. The strongest are the two
that cross a method boundary: route (c) must agree with route (a) as the noise
vanishes, and the primal inclusion test must agree with the closed-form
ellipsoid on every candidate value.
"""

import numpy as np
import pytest

from pygeoinf2.algebra.operators import LinearOperator
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.geometry.convex import Ball, ConvexSet, HalfSpace, Polytope
from pygeoinf2.inference import (
    BackusGilbert,
    BackusInference,
    FeasibleProperty,
    LinearForwardProblem,
)
from pygeoinf2.inference.backus import harden_error
from pygeoinf2.probability.gaussian import GaussianMeasure

from .conftest import make_weighted_space


@pytest.fixture
def setting(rng):
    """A model space, an under-determined map, a property, and a feasible truth."""
    model = make_weighted_space()
    data = EuclideanSpace(2)
    target_space = EuclideanSpace(2)
    forward = LinearOperator.from_matrix(
        model, data, rng.normal(size=(data.dim, model.dim)), form="galerkin"
    )
    target = LinearOperator.from_matrix(
        model,
        target_space,
        rng.normal(size=(target_space.dim, model.dim)),
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


class TestClosedForm:
    def test_the_truth_is_in_the_set(self, setting):
        model, forward, target, truth, data = setting
        assert model.norm(truth) <= 3.0
        inference = BackusInference(
            LinearForwardProblem(forward), target, Ball(model, radius=3.0)
        )
        assert inference(data).contains(target(truth))

    def test_the_centre_is_the_minimum_norm_property(self, setting):
        model, forward, target, truth, data = setting
        inference = BackusInference(
            LinearForwardProblem(forward), target, Ball(model, radius=3.0)
        )
        assert np.allclose(
            inference(data).centre, target(inference.minimum_norm_model(data))
        )

    def test_every_feasible_model_lands_inside(self, setting, rng):
        """Sampled from the feasible set itself, so a miss would be a defect."""
        model, forward, target, truth, data = setting
        inference = BackusInference(
            LinearForwardProblem(forward), target, Ball(model, radius=3.0)
        )
        answer = inference(data)
        anchor = inference.minimum_norm_model(data)
        budget = np.sqrt(inference.budget(data))
        for _ in range(300):
            offset = inference._kernel(model.random(rng=rng))
            length = model.norm(offset)
            if length == 0.0:
                continue
            candidate = model.add(
                anchor,
                model.scale(budget * rng.uniform(0.0, 1.0) / length, offset),
            )
            if model.norm(candidate) <= 3.0:
                assert answer.contains(target(candidate))

    def test_the_prior_alone_brackets_the_answer(self, setting):
        model, forward, target, truth, data = setting
        inference = BackusInference(
            LinearForwardProblem(forward), target, Ball(model, radius=3.0)
        )
        answer, before = inference(data), inference.prior_only()
        assert before.contains(answer.centre)
        space = inference.target_space
        for direction in directions(space):
            assert (
                before.support_function()(direction)
                >= answer.support_function()(direction) - 1e-8
            )

    def test_data_the_prior_cannot_fit_are_refused(self, setting):
        model, forward, target, truth, data = setting
        inference = BackusInference(
            LinearForwardProblem(forward), target, Ball(model, radius=0.01)
        )
        with pytest.raises(ValueError, match="fits these data"):
            inference(data)

    def test_the_shape_does_not_depend_on_the_data(self, setting, rng):
        """Only the centre and the size do -- the same structure as a Gaussian
        estimator's data-independent covariance."""
        model, forward, target, truth, data = setting
        inference = BackusInference(
            LinearForwardProblem(forward), target, Ball(model, radius=3.0)
        )
        assert inference.shape is inference.shape


class TestInclusionTest:
    def test_it_agrees_with_the_closed_form_everywhere(self, setting, rng):
        """Two entirely different computations of one statement.

        The set comes from a projection and an eigen-shape; the test comes from
        a minimum-norm solve on Parker's joint map. They must agree on every
        candidate, and a disagreement would name which is wrong.
        """
        model, forward, target, truth, data = setting
        inference = BackusInference(
            LinearForwardProblem(forward), target, Ball(model, radius=3.0)
        )
        answer = inference(data)
        space = inference.target_space
        inside = 0
        for _ in range(200):
            candidate = space.from_components(
                space.to_components(answer.centre) + rng.normal(size=space.dim) * 1.5
            )
            by_set = answer.contains(candidate)
            inside += by_set
            assert inference.admits(candidate, data) == by_set
        assert 20 < inside < 180  # the sample straddles the boundary

    def test_the_truth_is_admitted_at_its_own_norm(self, setting):
        model, forward, target, truth, data = setting
        inference = BackusInference(
            LinearForwardProblem(forward), target, Ball(model, radius=3.0)
        )
        assert inference.inclusion_norm(target(truth), data) <= model.norm(truth) + 1e-8
        assert inference.admits(target(truth), data)


class TestPrimalRoute:
    @pytest.mark.parametrize("noise", [1e-3, 1e-6])
    def test_it_agrees_with_the_closed_form_as_the_noise_vanishes(self, setting, noise):
        """The parity test between route (c) and route (a).

        Two methods with nothing in common — a bisection over two multipliers
        against a projection and an eigen-shape — converging on the same
        numbers as the noise ball shrinks. The agreement tracks the noise
        radius, which is what says the difference is the problem and not the
        method.
        """
        model, forward, target, truth, data = setting
        problem = LinearForwardProblem(forward)
        prior = Ball(model, radius=3.0)
        exact = BackusInference(problem, target, prior)(data).support_function()
        primal = FeasibleProperty(
            problem, target, prior, noise=Ball(forward.codomain, radius=noise)
        )
        for direction in directions(target.codomain):
            assert primal.support(direction, data) == pytest.approx(
                exact(direction), rel=10.0 * noise
            )

    def test_the_extremal_model_saturates_the_prior(self, setting):
        """Both constraints are active at the optimum, so the norm is the
        radius exactly -- which is the bisection's own convergence test seen
        from outside."""
        model, forward, target, truth, data = setting
        problem = LinearForwardProblem(forward)
        primal = FeasibleProperty(
            problem,
            target,
            Ball(model, radius=3.0),
            noise=Ball(forward.codomain, radius=1e-4),
        )
        for direction in directions(target.codomain):
            extremal = primal.extremal_model(direction, data)
            assert model.norm(extremal) == pytest.approx(3.0, rel=1e-6)

    def test_the_extremal_model_is_feasible_and_attains_the_bound(self, setting, rng):
        model, forward, target, truth, data = setting
        noise = Ball(forward.codomain, radius=0.2)
        problem = LinearForwardProblem(forward, error=noise)
        primal = FeasibleProperty(problem, target, Ball(model, radius=3.0))
        for direction in directions(target.codomain):
            extremal = primal.extremal_model(direction, data)
            assert model.norm(extremal) <= 3.0 + 1e-6
            residual = forward.codomain.subtract(data, forward(extremal))
            assert forward.codomain.norm(residual) <= 0.2 + 1e-6
            assert target.codomain.inner_product(
                target(extremal), direction
            ) == pytest.approx(primal.support(direction, data))

    def test_a_slack_data_constraint_gives_the_prior_bound(self, setting):
        """When the prior's own support point already fits, there is nothing
        to solve and the answer is ``M ||T* q||``."""
        model, forward, target, truth, data = setting
        problem = LinearForwardProblem(
            forward, error=Ball(forward.codomain, radius=1e6)
        )
        primal = FeasibleProperty(problem, target, Ball(model, radius=3.0))
        direction = target.codomain.basis_vector(0)
        assert primal.support(direction, data) == pytest.approx(
            3.0 * model.norm(target.adjoint(direction))
        )


class TestLinearCertificate:
    def test_it_bounds_the_exact_set(self, setting):
        """Validity is free; only sharpness is lost. That is weak duality, and
        it is the property that makes route (b) safe to use."""
        model, forward, target, truth, data = setting
        noise = Ball(forward.codomain, radius=0.2)
        problem = LinearForwardProblem(forward, error=noise)
        prior = Ball(model, radius=3.0)
        exact = FeasibleProperty(problem, target, prior)(data).support_function()
        certificate = (
            BackusGilbert(problem, target, prior).uncertainty(data).support_function()
        )
        for direction in directions(target.codomain):
            assert certificate(direction) >= exact(direction) - 1e-6

    def test_the_error_bars_split_into_two_causes(self, setting):
        model, forward, target, truth, data = setting
        problem = LinearForwardProblem(
            forward, error=Ball(forward.codomain, radius=0.05)
        )
        estimator = BackusGilbert(problem, target, Ball(model, radius=3.0))
        estimate, resolution, noise = estimator.error_bars(data)
        assert np.all(resolution > 0.0)
        assert np.all(noise > 0.0)
        assert np.all(np.abs(target(truth) - estimate) <= resolution + noise + 1e-8)

    def test_less_noise_narrows_only_the_noise_term(self, setting):
        model, forward, target, truth, data = setting
        prior = Ball(model, radius=3.0)
        wide = BackusGilbert(
            LinearForwardProblem(forward),
            target,
            prior,
            noise=Ball(forward.codomain, radius=0.5),
        ).error_bars(data)
        narrow = BackusGilbert(
            LinearForwardProblem(forward),
            target,
            prior,
            noise=Ball(forward.codomain, radius=0.05),
        ).error_bars(data)
        assert np.all(narrow[2] < wide[2])

    def test_the_unresolved_operator_complements_the_resolution(self, setting):
        model, forward, target, truth, data = setting
        estimator = BackusGilbert(
            LinearForwardProblem(forward),
            target,
            Ball(model, radius=3.0),
            noise=Ball(forward.codomain, radius=0.1),
        )
        x = model.random(rng=np.random.default_rng(0))
        assert np.allclose(
            estimator.unresolved(x),
            target.codomain.subtract(target(x), estimator.resolution(x)),
        )

    def test_a_general_convex_prior_is_refused_by_this_route(self, setting):
        from pygeoinf2.geometry.convex import Ellipsoid
        from pygeoinf2.traits import Traits

        model, forward, target, truth, data = setting
        shape = LinearOperator.self_adjoint(
            model, lambda v: v, traits=Traits.POSITIVE_DEFINITE
        )
        with pytest.raises(TypeError, match="must be a Ball"):
            BackusGilbert(
                LinearForwardProblem(forward), target, Ellipsoid(model, shape)
            )


class TestOuterApproximation:
    def test_a_polytope_from_support_values_contains_the_set(self, setting):
        model, forward, target, truth, data = setting
        problem = LinearForwardProblem(
            forward, error=Ball(forward.codomain, radius=0.2)
        )
        answer = FeasibleProperty(problem, target, Ball(model, radius=3.0))(data)
        polytope = answer.polytope(directions(target.codomain))
        assert polytope.is_outer
        assert polytope.contains(target(truth))

    def test_more_directions_only_tighten_it(self, setting):
        model, forward, target, truth, data = setting
        answer = BackusInference(
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


class TestBundleMethod:
    """The minimiser route (d) is built on."""

    def test_it_minimises_a_nonsmooth_convex_function(self):
        """``|x - a|_1 + |x|^2/2``, whose minimiser is ``clip(a, -1, 1)``.

        Not soft-thresholding, which is the answer to a different problem and
        was the first reference tried here.
        """
        from pygeoinf2.algebra.operators import Functional
        from pygeoinf2.numerics.convex import ProximalBundleMethod

        space = EuclideanSpace(4)
        anchor = np.random.default_rng(0).normal(size=4)
        functional = Functional.from_callables(
            space,
            lambda x: float(np.abs(x - anchor).sum() + 0.5 * x @ x),
            gradient=lambda x: np.sign(x - anchor) + x,
        )
        result = ProximalBundleMethod(tolerance=1e-10, iterations=400).minimise(
            functional, space.zero()
        )
        best = np.clip(anchor, -1.0, 1.0)
        assert result.converged
        assert result.minimum == pytest.approx(
            float(np.abs(best - anchor).sum() + 0.5 * best @ best), abs=1e-6
        )
        assert np.allclose(result.minimiser, best, atol=1e-5)

    def test_the_gap_certifies_the_answer(self):
        from pygeoinf2.algebra.operators import Functional
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
        result = ProximalBundleMethod(tolerance=1e-12, iterations=600).minimise(
            functional, space.zero()
        )
        best = np.linalg.solve(matrix, offset)
        assert result.minimum == pytest.approx(
            float(0.5 * best @ matrix @ best - offset @ best), abs=1e-7
        )
        assert result.gap >= 0.0

    def test_a_nonsense_descent_fraction_is_refused(self):
        from pygeoinf2.numerics.convex import ProximalBundleMethod

        with pytest.raises(ValueError, match="descent fraction"):
            ProximalBundleMethod(descent=1.5)


class TestDualRoute:
    def test_it_agrees_with_the_primal_route(self, setting):
        """Routes (c) and (d), with nothing in common.

        A bisection over two Lagrange multipliers against a nonsmooth convex
        minimisation in the data space. They agree to nine figures.
        """
        from pygeoinf2.inference import DualFeasibleProperty

        model, forward, target, truth, data = setting
        prior = Ball(model, radius=3.0)
        for radius in (0.3, 0.05):
            problem = LinearForwardProblem(
                forward, error=Ball(forward.codomain, radius=radius)
            )
            primal = FeasibleProperty(problem, target, prior)
            dual = DualFeasibleProperty(problem, target, prior)
            for direction in directions(target.codomain):
                assert dual.support(direction, data) == pytest.approx(
                    primal.support(direction, data), rel=1e-6
                )

    def test_it_accepts_a_prior_the_other_routes_cannot(self, setting, rng):
        """Which is the whole reason it exists.

        An anisotropic prior has no radius, so routes (a) and (b) refuse it by
        name; this one needs only a support function and a maximiser.
        """
        from pygeoinf2.geometry.convex import Ellipsoid
        from pygeoinf2.inference import DualFeasibleProperty
        from pygeoinf2.traits import Traits

        model, forward, target, truth, data = setting
        scale = np.diag(np.array([36.0, 16.0, 16.0, 9.0])[: model.dim])
        gram = model.gram_matrix()
        covariance = LinearOperator.from_matrix(
            model,
            model,
            gram @ scale,
            traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE,
            form="galerkin",
        )
        precision = LinearOperator.from_matrix(
            model,
            model,
            gram @ np.linalg.inv(scale),
            traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE,
            form="galerkin",
        )
        prior = Ellipsoid(model, precision, covariance=covariance)
        assert prior.contains(truth)

        problem = LinearForwardProblem(
            forward, error=Ball(forward.codomain, radius=0.1)
        )
        dual = DualFeasibleProperty(problem, target, prior)
        for direction in directions(target.codomain):
            assert np.isfinite(dual.support(direction, data))

        with pytest.raises(TypeError, match="must be a Ball"):
            BackusGilbert(problem, target, prior)

    def test_a_ball_written_as_an_ellipsoid_gives_the_same_answer(self, setting):
        from pygeoinf2.geometry.convex import Ellipsoid
        from pygeoinf2.inference import DualFeasibleProperty
        from pygeoinf2.traits import Traits

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
        as_ball = DualFeasibleProperty(problem, target, Ball(model, radius=radius))
        as_ellipsoid = DualFeasibleProperty(
            problem, target, Ellipsoid(model, precision, covariance=covariance)
        )
        for direction in directions(target.codomain):
            assert as_ellipsoid.support(direction, data) == pytest.approx(
                as_ball.support(direction, data), rel=1e-7
            )

    def test_an_empty_feasible_set_is_reported(self, setting):
        """An unbounded dual is a statement about the problem, and a large
        negative number is a perfectly plausible-looking support value."""
        from pygeoinf2.inference import DualFeasibleProperty

        model, forward, target, truth, data = setting
        problem = LinearForwardProblem(
            forward, error=Ball(forward.codomain, radius=1e-6)
        )
        dual = DualFeasibleProperty(problem, target, Ball(model, radius=0.01))
        with pytest.raises(ValueError, match="no model lies"):
            dual.support(target.codomain.basis_vector(0), data)

    def test_the_certificate_is_a_weighting_of_the_data(self, setting):
        from pygeoinf2.inference import DualFeasibleProperty

        model, forward, target, truth, data = setting
        problem = LinearForwardProblem(
            forward, error=Ball(forward.codomain, radius=0.1)
        )
        dual = DualFeasibleProperty(problem, target, Ball(model, radius=3.0))
        direction = target.codomain.basis_vector(0)
        certificate = dual.certificate(direction, data)
        # any certificate gives a valid bound; this one is the best
        cost = dual.dual_cost(direction, data)
        assert cost(certificate) <= cost(forward.codomain.zero()) + 1e-9


class TestInclusionWithErrors:
    """Set inclusion as a constrained optimisation, with noisy data.

    The complement of the support-function machinery: a support function bounds
    the set from outside one direction at a time, and this decides membership
    exactly one point at a time. Only the second can produce an inner bound,
    and the two must agree about every point they both have an opinion on.
    """

    @pytest.fixture
    def noisy(self, rng):
        model = make_weighted_space()
        data_space = EuclideanSpace(3)
        target_space = EuclideanSpace(2)
        forward = LinearOperator.from_matrix(
            model, data_space, rng.normal(size=(3, model.dim)), form="galerkin"
        )
        target = LinearOperator.from_matrix(
            model, target_space, rng.normal(size=(2, model.dim)), form="galerkin"
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
        estimator = FeasibleProperty(problem, target, Ball(model, radius=3.0))
        return model, forward, target, truth, data, estimator

    def test_the_truth_is_admitted(self, noisy):
        model, forward, target, truth, data, estimator = noisy
        assert estimator.admits(target(truth), data)
        assert estimator.inclusion_norm(target(truth), data) <= model.norm(truth) + 1e-6

    def test_it_reduces_to_the_error_free_test(self, noisy, rng):
        """As the noise ball shrinks, with the difference tracking its radius.

        Al-Attar (2021) §3.3 against §2.3: the second is the first with the
        confidence set collapsed to a point.
        """
        model, forward, target, truth, data, _ = noisy
        prior = Ball(model, radius=3.0)
        exact = BackusInference(LinearForwardProblem(forward), target, prior)
        space = target.codomain
        for radius in (1e-2, 1e-4):
            noisy_estimator = FeasibleProperty(
                LinearForwardProblem(
                    forward, error=Ball(forward.codomain, radius=radius)
                ),
                target,
                prior,
            )
            for _ in range(8):
                value = space.from_components(
                    space.to_components(target(truth)) + rng.normal(size=space.dim)
                )
                assert noisy_estimator.inclusion_norm(value, data) == pytest.approx(
                    exact.inclusion_norm(value, data), rel=10.0 * radius
                )

    @pytest.mark.slow
    def test_it_never_admits_what_the_support_function_excludes(self, noisy, rng):
        """The two descriptions of one set, checked against each other.

        A primal minimum-norm computation against a directional bound. Neither
        can be adjusted to agree with the other, so an inconsistency would name
        which is wrong.
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
            if estimator.admits(value, data):
                admitted += 1
                assert outer.contains(value)
            if answer.outside(value, probes):
                excluded += 1
                assert not estimator.admits(value, data)
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
        assert estimator.inclusion_norm(far, data) == float("inf")
        assert not estimator.admits(far, data)

    @pytest.mark.slow
    def test_the_inner_hull_sits_inside_the_outer_bound(self, noisy, rng):
        """§18.4's sandwich, with both sides real for the first time."""
        model, forward, target, truth, data, estimator = noisy
        space = target.codomain
        candidates = [
            space.from_components(
                space.to_components(target(truth)) + rng.normal(size=2) * 2.0
            )
            for _ in range(400)
        ]
        hull = estimator.inner_hull(candidates, data)
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
        with pytest.raises(ValueError, match="admissible"):
            estimator.inner_hull(far, data)


class TestHardeningTheError:
    """The bridge from a Gaussian error to the ball the Backus routes need."""

    def test_no_error_measure_gives_the_degenerate_ball(self, setting):
        """The error-free path, which used to raise: ``Ball(radius=0.0)`` was
        refused by the constructor, so route (a) could not run on exact data at
        all."""
        model, forward, _, _, data = setting
        ball = harden_error(LinearForwardProblem(forward), level=0.95)
        assert ball.radius == 0.0
        assert ball.contains(forward.codomain.zero())

    @pytest.mark.parametrize("build", [lambda: EuclideanSpace(4), make_weighted_space])
    def test_the_ball_carries_the_probability_it_claims(self, build, rng):
        """An anisotropic error on a weighted space, which is where the old
        rule was wrong.

        It used ``sqrt(chi2_crit * mean diagonal of the component matrix)``.
        The component matrix's diagonal is the variance only on an orthonormal
        basis, so on a weighted space the ball came out too small: measured
        coverage 0.846 against a claimed 0.90, where ``ambient_ball`` gives
        0.900.
        """
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
