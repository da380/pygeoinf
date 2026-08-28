"""Nonlinear operators, functionals, and the evaluation paths."""

import numpy as np
import pytest

from pygeoinf2.algebra.linearisation import Linearisation, QuadraticModel
from pygeoinf2.algebra.operators import (
    AffineOperator,
    Functional,
    LinearFunctional,
    LinearOperator,
    Operator,
)
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.testing import (
    check_derivative,
    check_gradient,
    check_second_derivative,
)

from .conftest import make_weighted_space
from .doubles import OpaqueSpace


def quadratic_on(space, matrix, rng=None):
    """phi(x) = 0.5 c_x . M c_x, with derivative and Hessian supplied."""
    matrix = 0.5 * (matrix + matrix.T)

    def value(x):
        c = space.to_components(x)
        return 0.5 * float(c @ matrix @ c)

    def derivative(x):
        # An adjoint solve returns THIS: components of dphi/dc.
        return LinearFunctional.from_derivative_components(
            space, matrix @ space.to_components(x)
        )

    def hessian(x):
        return LinearOperator.self_adjoint(
            space,
            lambda v: space.from_components(
                space.solve_gram(matrix @ space.to_components(v))
            ),
        )

    return Functional.from_callables(
        space, value, derivative=derivative, hessian=hessian
    )


class TestFunctional:
    def test_gradient_is_the_representer_not_the_derivative(self, rng):
        """The core of DESIGN.md 5.6, on a space where it is visible."""
        X = make_weighted_space()
        M = rng.normal(size=(X.dim, X.dim))
        phi = quadratic_on(X, M)
        x = X.random(rng=rng)

        model = phi.at(x)
        derivative_components = model.derivative.matrix().ravel()
        gradient_components = X.to_components(model.gradient)

        # The gradient is the derivative with the inverse metric applied.
        assert np.allclose(gradient_components, X.solve_gram(derivative_components))
        assert not np.allclose(gradient_components, derivative_components)
        check_gradient(phi, x, rng=rng)

    def test_the_check_catches_a_derivative_supplied_as_a_gradient(self, rng):
        """The classic adjoint-method error, made into a test failure."""
        X = make_weighted_space()
        M = 0.5 * (lambda A: A + A.T)(rng.normal(size=(X.dim, X.dim)))

        def value(x):
            c = X.to_components(x)
            return 0.5 * float(c @ M @ c)

        def wrong_gradient(x):
            # Handing back the derivative components as if they were a vector.
            return X.from_components(M @ X.to_components(x))

        phi = Functional.from_callables(X, value, gradient=wrong_gradient)
        with pytest.raises(AssertionError, match="gradient pairs with a direction"):
            check_gradient(phi, X.random(rng=rng), rng=rng)

    def test_the_error_is_invisible_on_an_orthonormal_space(self, rng):
        """Which is why it survives: the toy case cannot detect it."""
        X = EuclideanSpace(4)
        M = 0.5 * (lambda A: A + A.T)(rng.normal(size=(4, 4)))

        def value(x):
            return 0.5 * float(x @ M @ x)

        phi = Functional.from_callables(X, value, gradient=lambda x: M @ x)
        check_gradient(phi, X.random(rng=rng), rng=rng)  # passes, correctly

    def test_at_returns_a_quadratic_model(self, rng):
        X = make_weighted_space()
        phi = quadratic_on(X, rng.normal(size=(X.dim, X.dim)))
        model = phi.at(X.random(rng=rng))
        assert isinstance(model, QuadraticModel)
        assert model.has_hessian
        assert isinstance(model.value, float)

    def test_gradient_is_cached(self, rng):
        X = make_weighted_space()
        phi = quadratic_on(X, rng.normal(size=(X.dim, X.dim)))
        model = phi.at(X.random(rng=rng))
        assert model.gradient is model.gradient

    def test_hessian_is_absent_unless_supplied(self, rng):
        X = EuclideanSpace(3)
        phi = Functional.from_callables(
            X, lambda x: float(x @ x), gradient=lambda x: 2.0 * x
        )
        assert not phi.has_hessian
        assert phi.at(X.random(rng=rng)).hessian is None

    def test_derivative_and_gradient_cannot_both_be_given(self):
        X = EuclideanSpace(3)
        with pytest.raises(ValueError, match="not both"):
            Functional.from_callables(
                X, lambda x: 0.0, derivative=lambda x: None, gradient=lambda x: x
            )


class TestLinearFunctional:
    def test_matrix_is_the_derivative_and_adjoint_is_the_gradient(self, rng):
        """Two readings of one operator; the adjoint carries the metric."""
        X = make_weighted_space()
        g = rng.normal(size=X.dim)
        f = LinearFunctional.from_derivative_components(X, g)

        assert np.allclose(f.matrix().ravel(), g)
        assert np.allclose(X.to_components(f.representer), X.solve_gram(g))
        assert f.representer is f.adjoint(1.0) or np.allclose(
            X.to_components(f.representer), X.to_components(f.adjoint(1.0))
        )

    def test_acts_as_the_derivative_pairing(self, rng):
        X = make_weighted_space()
        g = rng.normal(size=X.dim)
        f = LinearFunctional.from_derivative_components(X, g)
        x = X.random(rng=rng)
        assert f(x) == pytest.approx(float(g @ X.to_components(x)))

    def test_from_representer_pairs_by_inner_product(self, rng):
        X = make_weighted_space()
        v = X.random(rng=rng)
        f = LinearFunctional.from_representer(X, v)
        x = X.random(rng=rng)
        assert f(x) == pytest.approx(X.inner_product(v, x))

    def test_works_without_coordinates(self, rng):
        """from_representer needs no component map at all."""
        X = OpaqueSpace(np.array([1.0, 2.0, 3.0]))
        v = X.random(rng=rng)
        f = LinearFunctional.from_representer(X, v)
        x = X.random(rng=rng)
        assert f(x) == pytest.approx(X.inner_product(v, x))

    def test_hessian_is_zero(self, rng):
        X = EuclideanSpace(3)
        f = LinearFunctional.from_derivative_components(X, np.array([1.0, 2.0, 3.0]))
        assert np.allclose(f.hessian(X.random(rng=rng))(X.random(rng=rng)), np.zeros(3))


class TestNonlinearOperator:
    def build(self, X, Y, rng):
        """F(x)_i = (c_x . B_i c_x), a genuinely quadratic map."""
        blocks = [rng.normal(size=(X.dim, X.dim)) for _ in range(Y.dim)]
        blocks = [0.5 * (B + B.T) for B in blocks]

        def value(x):
            c = X.to_components(x)
            return Y.from_components(np.array([c @ B @ c for B in blocks]))

        def derivative(x):
            c = X.to_components(x)
            return LinearOperator.from_matrix(
                X, Y, np.stack([2.0 * (B @ c) for B in blocks]), form="components"
            )

        def second_derivative(x, dx):
            cd = X.to_components(dx)
            return LinearOperator.from_matrix(
                X, Y, np.stack([2.0 * (B @ cd) for B in blocks]), form="components"
            )

        return Operator.from_callables(
            X, Y, value, derivative=derivative, second_derivative=second_derivative
        )

    def test_derivative(self, rng):
        X, Y = make_weighted_space(), EuclideanSpace(2)
        F = self.build(X, Y, rng)
        check_derivative(F, X.random(rng=rng), rng=rng)

    def test_second_derivative(self, rng):
        X, Y = make_weighted_space(), EuclideanSpace(2)
        F = self.build(X, Y, rng)
        check_second_derivative(F, X.random(rng=rng), rng=rng)

    def test_chain_rule(self, rng):
        X, Y, Z = make_weighted_space(), EuclideanSpace(3), EuclideanSpace(2)
        F = self.build(X, Y, rng)
        A = LinearOperator.from_matrix(Y, Z, rng.normal(size=(2, 3)), form="components")
        composed = A @ F
        assert composed.has_derivative
        check_derivative(composed, X.random(rng=rng), rng=rng)

    def test_second_derivative_survives_composition(self, rng):
        """(F o G)'' needs both factors to carry one."""
        X, Y, Z = make_weighted_space(), EuclideanSpace(3), EuclideanSpace(2)
        F = self.build(X, Y, rng)
        G = self.build(Y, Z, rng)
        composed = G @ F
        assert composed.has_second_derivative
        check_second_derivative(composed, X.random(rng=rng), rng=rng, rtol=1e-3)

    def test_second_derivative_absent_when_a_factor_lacks_one(self, rng):
        X, Y, Z = make_weighted_space(), EuclideanSpace(3), EuclideanSpace(2)
        F = self.build(X, Y, rng)
        partial = Operator.from_callables(
            Y,
            Z,
            lambda y: np.array([y @ y, 0.0]),
            derivative=lambda y: LinearOperator.from_matrix(
                Y, Z, np.stack([2.0 * y, np.zeros(3)]), form="components"
            ),
        )
        assert not (partial @ F).has_second_derivative

    def test_sum_and_scaling(self, rng):
        X, Y = make_weighted_space(), EuclideanSpace(2)
        F = self.build(X, Y, rng)
        G = self.build(X, Y, rng)
        for op in (F + G, 2.5 * F, F - G):
            check_derivative(op, X.random(rng=rng), rng=rng)


class TestSharedWork:
    """`at()` exists so that value and derivative share one evaluation."""

    def test_at_evaluates_once(self, rng):
        X, Y = EuclideanSpace(3), EuclideanSpace(2)
        calls = {"n": 0}
        M = rng.normal(size=(2, 3))

        def linearise(x):
            calls["n"] += 1  # stands in for the PDE solve
            value = Y.from_components(M @ X.to_components(x))
            return Linearisation(
                x, value, LinearOperator.from_matrix(X, Y, M, form="components")
            )

        F = Operator.from_callables(X, Y, lambda x: M @ x, linearise=linearise)

        model = F.at(X.random(rng=rng))
        _ = model.value, model.derivative
        assert calls["n"] == 1, "at() should linearise once, not once per accessor"

    def test_call_does_not_linearise(self, rng):
        """A line search does many value-only evaluations and must not pay."""
        X, Y = EuclideanSpace(3), EuclideanSpace(2)
        calls = {"n": 0}
        M = rng.normal(size=(2, 3))

        def linearise(x):
            calls["n"] += 1
            return Linearisation(
                x,
                Y.from_components(M @ X.to_components(x)),
                LinearOperator.from_matrix(X, Y, M, form="components"),
            )

        F = Operator.from_callables(X, Y, lambda x: M @ x, linearise=linearise)
        for _ in range(20):
            F(X.random(rng=rng))
        assert calls["n"] == 0


class TestAffineOperator:
    def test_value_and_derivative(self, rng):
        X, Y = make_weighted_space(), EuclideanSpace(3)
        A = LinearOperator.from_matrix(
            X, Y, rng.normal(size=(3, X.dim)), form="components"
        )
        b = Y.random(rng=rng)
        F = AffineOperator(A, b)
        x = X.random(rng=rng)
        assert np.allclose(F(x), A(x) + b)
        assert F.derivative(x) is A
        check_derivative(F, x, rng=rng)

    def test_affineness_survives_the_algebra(self, rng):
        """v1 preserves this with a string type check; the protocol replaces it."""
        X, Y = EuclideanSpace(4), EuclideanSpace(3)
        A = LinearOperator.from_matrix(X, Y, rng.normal(size=(3, 4)), form="components")
        B = LinearOperator.from_matrix(X, Y, rng.normal(size=(3, 4)), form="components")
        F = AffineOperator(A, Y.random(rng=rng))

        assert isinstance(F + B, AffineOperator)
        assert isinstance(B + F, AffineOperator)  # the order-independent case
        assert isinstance(2.0 * F, AffineOperator)
        assert isinstance(F + F, AffineOperator)

    def test_composition_with_linear_operators_stays_affine(self, rng):
        X, Y, Z = EuclideanSpace(4), EuclideanSpace(3), EuclideanSpace(2)
        A = LinearOperator.from_matrix(X, Y, rng.normal(size=(3, 4)), form="components")
        C = LinearOperator.from_matrix(Y, Z, rng.normal(size=(2, 3)), form="components")
        D = LinearOperator.from_matrix(Z, X, rng.normal(size=(4, 2)), form="components")
        F = AffineOperator(A, Y.random(rng=rng))

        assert isinstance(C @ F, AffineOperator)
        assert isinstance(F @ D, AffineOperator)
        x = Z.random(rng=rng)
        assert np.allclose((C @ F)(D(x)), C(F(D(x))))

    def test_linearisation_as_affine(self, rng):
        X, Y = make_weighted_space(), EuclideanSpace(2)
        F = TestNonlinearOperator().build(X, Y, rng)
        x = X.random(rng=rng)
        model = F.at(x)
        affine = model.as_affine()
        assert np.allclose(affine(x), model.value)


class TestTheFunctionalAlgebraIsClosed:
    """A sum of functionals is a functional, and so is a composition.

    It was not. ``phi + psi`` came back as a plain ``Operator``: no
    ``.gradient``, no ``.hessian``, and ``at(x)`` a linearisation rather than a
    quadratic model. So ``functional.at(x).gradient``, which is the first thing
    every optimiser here asks for, raised ``AttributeError`` on any composed
    objective -- which is to say on every real one, since a misfit plus a
    regulariser is a sum.
    """

    @pytest.fixture(params=["euclidean", "weighted"])
    def space(self, request):
        return (
            EuclideanSpace(3) if request.param == "euclidean" else make_weighted_space()
        )

    def quadratic(self, space, matrix):
        return Functional.from_callables(
            space,
            lambda x: 0.5
            * float(space.to_components(x) @ matrix @ space.to_components(x)),
            gradient=lambda x: space.from_components(
                space.solve_gram(matrix @ space.to_components(x))
            ),
            hessian=lambda x: LinearOperator.from_matrix(
                space, space, matrix, form="galerkin"
            ),
        )

    def test_a_sum_is_a_functional_with_a_gradient(self, space, rng):
        matrix = np.identity(space.dim)
        phi = self.quadratic(space, matrix)
        psi = self.quadratic(space, 2.0 * matrix)
        total = phi + psi

        assert isinstance(total, Functional)
        assert isinstance(total.at(space.zero()), QuadraticModel)
        probe = space.random(rng=rng)
        expected = space.add(phi.gradient(probe), psi.gradient(probe))
        assert space.norm(
            space.subtract(total.at(probe).gradient, expected)
        ) < 1e-10 * space.norm(expected)

    def test_a_scaling_scales_the_gradient_and_the_hessian(self, space, rng):
        phi = self.quadratic(space, np.identity(space.dim))
        scaled = 3.0 * phi
        probe = space.random(rng=rng)

        assert isinstance(scaled, Functional)
        assert (
            space.norm(
                space.subtract(
                    scaled.gradient(probe), space.scale(3.0, phi.gradient(probe))
                )
            )
            < 1e-10
        )
        assert scaled.hessian(probe).matrix(form="galerkin") == pytest.approx(
            3.0 * phi.hessian(probe).matrix(form="galerkin")
        )

    def test_a_hessian_needs_every_term_to_have_one(self, space):
        """Reported rather than assumed: a sum with one Hessian-less term has
        none, and says so instead of failing when asked."""
        phi = self.quadratic(space, np.identity(space.dim))
        flat = Functional.from_callables(
            space, lambda x: 0.0, gradient=lambda x: space.zero()
        )
        assert phi.has_hessian
        assert not flat.has_hessian
        assert not (phi + flat).has_hessian

    def test_composing_with_a_linear_map_gives_the_congruence(self, rng):
        """``phi @ A`` has Hessian ``A* H A``, exactly."""
        domain, codomain = EuclideanSpace(3), EuclideanSpace(2)
        matrix = rng.normal(size=(2, 3))
        operator = LinearOperator.from_matrix(
            domain, codomain, matrix, form="components"
        )
        phi = Functional.from_callables(
            codomain,
            lambda y: 0.5 * float(y @ y),
            gradient=lambda y: y,
            hessian=lambda y: LinearOperator.identity(codomain),
        )
        composed = phi @ operator

        assert isinstance(composed, Functional)
        probe = rng.normal(size=3)
        assert composed.at(probe).gradient == pytest.approx(matrix.T @ (matrix @ probe))
        assert composed.hessian(probe).matrix(form="components") == pytest.approx(
            matrix.T @ matrix
        )

    def test_composing_with_a_curved_map_keeps_the_curvature_term(self, rng):
        """The Gauss-Newton term is not the whole Hessian.

        ``(phi o F)'' == F'* H F' + F''[.]* grad``, and the second term is what
        makes the Hessian indefinite away from a minimum. Checked against
        central differences of the gradient, which knows nothing of either.
        """
        domain = codomain = EuclideanSpace(2)

        def value(x):
            return np.array([x[0] ** 2, x[0] * x[1]])

        def derivative(x):
            return LinearOperator.from_matrix(
                domain,
                codomain,
                np.array([[2.0 * x[0], 0.0], [x[1], x[0]]]),
                form="components",
            )

        def second(x, dx):
            return LinearOperator.from_matrix(
                domain,
                codomain,
                np.array([[2.0 * dx[0], 0.0], [dx[1], dx[0]]]),
                form="components",
            )

        curved = Operator.from_callables(
            domain, codomain, value, derivative=derivative, second_derivative=second
        )
        phi = Functional.from_callables(
            codomain,
            lambda y: 0.5 * float(y @ y),
            gradient=lambda y: y,
            hessian=lambda y: LinearOperator.identity(codomain),
        )
        composed = phi @ curved

        point = np.array([1.3, -0.7])
        analytic = composed.hessian(point).matrix(form="components")

        step = 1e-6
        numerical = np.zeros((2, 2))
        for column in range(2):
            shift = np.zeros(2)
            shift[column] = step
            forward = composed.gradient(point + shift)
            backward = composed.gradient(point - shift)
            numerical[:, column] = (forward - backward) / (2.0 * step)
        assert analytic == pytest.approx(numerical, abs=1e-6)

    def test_an_optimiser_runs_on_a_composed_objective(self, rng):
        """The payoff, and the thing that used to raise. A misfit plus a
        regulariser is a sum of a composition and a scaling."""
        from pygeoinf2.numerics.optimisation import LBFGS, NewtonCG

        domain, data_space = EuclideanSpace(12), EuclideanSpace(20)
        matrix = rng.normal(size=(20, 12))
        observed = rng.normal(size=20)
        forward = LinearOperator.from_matrix(
            domain, data_space, matrix, form="components"
        )
        misfit = Functional.from_callables(
            data_space,
            lambda y: 0.5 * float((y - observed) @ (y - observed)),
            gradient=lambda y: y - observed,
            hessian=lambda y: LinearOperator.identity(data_space),
        )
        penalty = Functional.from_callables(
            domain,
            lambda x: 0.5 * float(x @ x),
            gradient=lambda x: x,
            hessian=lambda x: LinearOperator.identity(domain),
        )
        objective = (misfit @ forward) + 0.1 * penalty

        exact = np.linalg.solve(
            matrix.T @ matrix + 0.1 * np.identity(12), matrix.T @ observed
        )
        for optimiser in (LBFGS(max_iterations=2000), NewtonCG(max_iterations=200)):
            result = optimiser.minimise(objective, domain.zero())
            assert result.converged, result.message
            assert result.minimiser == pytest.approx(exact, rel=1e-4)


class TestTheLinearFunctionalAlgebraIsClosed:
    """``f + g``, ``2 f`` and ``f @ A`` are linear functionals too."""

    @pytest.fixture(params=["euclidean", "dense-metric"])
    def space(self, request):
        from .conftest import make_dense_metric_space

        return (
            EuclideanSpace(3)
            if request.param == "euclidean"
            else make_dense_metric_space()
        )

    def test_the_representer_survives(self, space, rng):
        """It is derivable from the adjoint, which the plain node has -- but it
        lives on ``LinearFunctional``, so losing the type loses the reading."""
        first = LinearFunctional.from_representer(space, space.random(rng=rng))
        second = LinearFunctional.from_representer(space, space.random(rng=rng))
        doubling = LinearOperator.from_matrix(
            space, space, 2.0 * np.identity(space.dim), form="components"
        )

        for derived in (first + second, 2.0 * first, first @ doubling):
            assert isinstance(derived, LinearFunctional)
            probe = space.random(rng=rng)
            # The defining property: f(x) == (representer, x).
            assert space.inner_product(derived.representer, probe) == pytest.approx(
                derived(probe)
            )
            # And the derivative components differ from it by the metric.
            assert derived.derivative_components == pytest.approx(
                space.apply_gram(space.to_components(derived.representer))
            )

    def test_it_can_be_built_from_a_mapping(self, space, rng):
        """v1's ``LinearForm(domain, mapping=...)`` had no counterpart: this
        name resolved by MRO to ``LinearOperator.from_callables``."""
        vector = space.random(rng=rng)
        by_representer = LinearFunctional.from_callables(
            space,
            lambda x: space.inner_product(vector, x),
            representer=lambda: vector,
        )
        by_components = LinearFunctional.from_callables(
            space,
            lambda x: float(
                space.apply_gram(space.to_components(vector)) @ space.to_components(x)
            ),
            derivative_components=lambda: space.apply_gram(space.to_components(vector)),
        )
        probe = space.random(rng=rng)
        for functional in (by_representer, by_components):
            assert isinstance(functional, LinearFunctional)
            assert functional(probe) == pytest.approx(
                space.inner_product(vector, probe)
            )
            assert space.norm(
                space.subtract(functional.representer, vector)
            ) < 1e-10 * space.norm(vector)

    def test_exactly_one_reading_is_needed(self, space):
        with pytest.raises(ValueError, match="exactly one"):
            LinearFunctional.from_callables(space, lambda x: 0.0)
        with pytest.raises(ValueError, match="exactly one"):
            LinearFunctional.from_callables(
                space,
                lambda x: 0.0,
                representer=lambda: space.zero(),
                derivative_components=lambda: np.zeros(space.dim),
            )
