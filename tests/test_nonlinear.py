"""Nonlinear operators, functionals, and the evaluation paths."""

import numpy as np
import pytest

from pygeoinf.algebra.linearization import Linearization, QuadraticModel
from pygeoinf.algebra.operators import (
    AffineOperator,
    Functional,
    LinearFunctional,
    LinearOperator,
    Operator,
)
from pygeoinf.algebra.spaces import EuclideanSpace
from pygeoinf.testing import (
    check_derivative,
    check_gradient,
    check_second_derivative,
)

from .conftest import make_dense_metric_space, make_weighted_space
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
        """A derivative is not a gradient, on a space where it is visible."""
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

        def linearize(x):
            calls["n"] += 1  # stands in for the PDE solve
            value = Y.from_components(M @ X.to_components(x))
            return Linearization(
                x, value, LinearOperator.from_matrix(X, Y, M, form="components")
            )

        F = Operator.from_callables(X, Y, lambda x: M @ x, linearize=linearize)

        model = F.at(X.random(rng=rng))
        _ = model.value, model.derivative
        assert calls["n"] == 1, "at() should linearize once, not once per accessor"

    def test_call_does_not_linearize(self, rng):
        """A line search does many value-only evaluations and must not pay."""
        X, Y = EuclideanSpace(3), EuclideanSpace(2)
        calls = {"n": 0}
        M = rng.normal(size=(2, 3))

        def linearize(x):
            calls["n"] += 1
            return Linearization(
                x,
                Y.from_components(M @ X.to_components(x)),
                LinearOperator.from_matrix(X, Y, M, form="components"),
            )

        F = Operator.from_callables(X, Y, lambda x: M @ x, linearize=linearize)
        for _ in range(20):
            F(X.random(rng=rng))
        assert calls["n"] == 0


class TestDerivativeAlone:
    """``derivative(x)`` never evaluates a value the caller did not ask for.

    Routed through ``at()``, a derivative-only query on a forward operator
    whose value is a PDE solve would pay for that solve too; the counts here
    are the minimal ones.
    """

    @staticmethod
    def spy(X, Y, M, calls, name):
        def value(x):
            calls[name + "v"] += 1
            return Y.from_components(M @ X.to_components(x))

        def derivative(x):
            calls[name + "d"] += 1
            return LinearOperator.from_matrix(X, Y, M, form="components")

        return Operator.from_callables(X, Y, value, derivative=derivative)

    def test_plain_scaled_and_summed_take_no_value(self, rng):
        from collections import Counter

        X, Y = EuclideanSpace(3), EuclideanSpace(3)
        calls = Counter()
        F = self.spy(X, Y, rng.normal(size=(3, 3)), calls, "F")
        x = X.random(rng=rng)
        for operator in (F, 2.0 * F, F + F, -F):
            operator.derivative(x)
        assert calls["Fv"] == 0
        assert calls["Fd"] == 5

    def test_a_composition_takes_the_inner_value_only(self, rng):
        from collections import Counter

        X = EuclideanSpace(3)
        calls = Counter()
        F = self.spy(X, X, rng.normal(size=(3, 3)), calls, "F")
        G = self.spy(X, X, rng.normal(size=(3, 3)), calls, "G")
        (F @ G).derivative(X.random(rng=rng))
        assert calls == {"Gv": 1, "Gd": 1, "Fd": 1}

    def test_a_linearize_only_operator_still_answers(self, rng):
        X, Y = EuclideanSpace(3), EuclideanSpace(2)
        M = rng.normal(size=(2, 3))
        calls = {"n": 0}

        def linearize(x):
            calls["n"] += 1
            return Linearization(
                x,
                Y.from_components(M @ X.to_components(x)),
                LinearOperator.from_matrix(X, Y, M, form="components"),
            )

        F = Operator.from_callables(X, Y, lambda x: M @ x, linearize=linearize)
        assert F.has_derivative
        x = X.random(rng=rng)
        assert np.allclose(F.derivative(x)(x), M @ X.to_components(x))
        assert calls["n"] == 1

    def test_no_derivative_still_says_so(self, rng):
        X = EuclideanSpace(3)
        F = Operator.from_callables(X, X, lambda x: x)
        assert not F.has_derivative
        with pytest.raises(NotImplementedError, match="derivative"):
            F.derivative(X.random(rng=rng))
        with pytest.raises(NotImplementedError, match="derivative"):
            (F + F).derivative(X.random(rng=rng))


class TestAffineCheck:
    """The affine axiom checks, ``testing.check_affine``: the translation
    is recovered at zero, affine combinations are preserved, the derivative
    is the linear part, and the linear part passes its own checks."""

    @pytest.fixture(params=[make_weighted_space, make_dense_metric_space])
    def spaces(self, request):
        return request.param(), EuclideanSpace(3)

    def test_a_correct_affine_operator_passes(self, spaces, rng):
        from pygeoinf.testing import check_affine

        X, Y = spaces
        A = LinearOperator.from_matrix(
            X, Y, rng.normal(size=(3, X.dim)), form="galerkin"
        )
        check_affine(AffineOperator(A, Y.random(rng=rng)), rng=rng)
        # A deferred translation is computed on demand and checked the same.
        check_affine(AffineOperator(A, lambda: Y.zero(), lazy=True), rng=rng)

    def test_a_wrong_derivative_or_translation_fails(self, spaces, rng):
        """Built by hand from the operator protocol, so the affine claim can
        be made false in one place at a time."""
        from pygeoinf.testing import check_affine

        X, Y = spaces
        A = LinearOperator.from_matrix(
            X, Y, rng.normal(size=(3, X.dim)), form="galerkin"
        )
        b = Y.random(rng=rng)

        class WrongDerivative(AffineOperator):
            # A derivative query goes through the linearization, so both
            # routes must lie for the claim to be testably false.
            def _derivative(self, x):
                return A * 2.0

            def _linearize(self, x):
                from pygeoinf.algebra.operators import Linearization

                return Linearization(x, self._value(x), A * 2.0)

        with pytest.raises(AssertionError, match="derivative is the linear part"):
            check_affine(WrongDerivative(A, b), rng=rng)

        class NotAffine(AffineOperator):
            def _value(self, x):
                return Y.scale(X.norm(x), self.linear_part(x))

        with pytest.raises(AssertionError, match="affine combinations|translation"):
            check_affine(NotAffine(A, b), rng=rng)

        with pytest.raises(TypeError, match="AffineOperator"):
            check_affine(A, rng=rng)

    def test_a_bad_adjoint_in_the_linear_part_is_caught(self, spaces, rng):
        """The linear part is checked too; a wrong adjoint on a non-orthonormal
        space is the classic mistake."""
        from pygeoinf.testing import check_affine

        X, Y = spaces
        matrix = rng.normal(size=(3, X.dim))
        wrong = LinearOperator.from_callables(
            X,
            Y,
            lambda x: Y.from_components(matrix @ X.to_components(x)),
            adjoint=lambda y: X.from_components(matrix.T @ Y.to_components(y)),
        )
        with pytest.raises(AssertionError, match="adjoint"):
            check_affine(AffineOperator(wrong, Y.random(rng=rng)), rng=rng)


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

    def test_a_callable_vector_is_a_translation_not_a_thunk(self, rng):
        """A backend's vectors may be callable, a field that evaluates at a
        point; the translation is taken as a vector unless ``lazy=True``."""
        from .doubles import Opaque

        class Field(Opaque):
            __slots__ = ()

            def __call__(self, point):
                return float(np.polyval(self.data, point))

        class FieldSpace(OpaqueSpace):
            def zero(self):
                return Field(np.zeros(self.dim))

            def copy(self, x):
                return Field(x.data.copy())

            def random(self, *, rng=None):
                return Field(rng.standard_normal(self.dim))

        V = FieldSpace(np.array([1.0, 2.0, 3.0]))
        b = Field(np.array([1.0, 2.0, 3.0]))
        F = AffineOperator(LinearOperator.identity(V), b)
        assert F.translation is b
        assert np.allclose(F(V.zero()).data, b.data)
        lazy = AffineOperator(LinearOperator.identity(V), lambda: b, lazy=True)
        assert lazy.translation is b

    def test_a_lazy_translation_must_be_callable(self, rng):
        Y = EuclideanSpace(3)
        with pytest.raises(TypeError, match="zero-argument callable"):
            AffineOperator(LinearOperator.identity(Y), Y.zero(), lazy=True)

    def test_affineness_survives_the_algebra(self, rng):
        """The protocol, not a type check, carries affineness through the algebra."""
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

    def test_linearization_as_affine(self, rng):
        X, Y = make_weighted_space(), EuclideanSpace(2)
        F = TestNonlinearOperator().build(X, Y, rng)
        x = X.random(rng=rng)
        model = F.at(x)
        affine = model.as_affine()
        assert np.allclose(affine(x), model.value)


class TestTheFunctionalAlgebraIsClosed:
    """A sum of functionals is a functional, and so is a composition.

    It was not. ``phi + psi`` came back as a plain ``Operator``: no
    ``.gradient``, no ``.hessian``, and ``at(x)`` a linearization rather than a
    quadratic model. So ``functional.at(x).gradient``, which is the first thing
    every optimizer here asks for, raised ``AttributeError`` on any composed
    objective -- which is to say on every real one, since a misfit plus a
    regularizer is a sum.
    """

    @pytest.fixture(params=["euclidean", "weighted", "dense"])
    def space(self, request):
        return {
            "euclidean": lambda: EuclideanSpace(3),
            "weighted": make_weighted_space,
            "dense": lambda: make_dense_metric_space(3),
        }[request.param]()

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

    def test_an_optimizer_runs_on_a_composed_objective(self, rng):
        """The payoff. A misfit plus a
        regularizer is a sum of a composition and a scaling."""
        from pygeoinf.numerics.optimization import LBFGS, NewtonCG

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
        for optimizer in (LBFGS(max_iterations=2000), NewtonCG(max_iterations=200)):
            result = optimizer.minimize(objective, domain.zero())
            assert result.converged, result.message
            assert result.minimizer == pytest.approx(exact, rel=1e-4)


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
        """A functional from a mapping, by this class's own constructor and
        not one resolved by MRO to ``LinearOperator.from_callables``."""
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


class TestTheFunctionalAlgebraLinearizesOnce:
    """A sum or a scaling of functionals builds its Hessian from the quadratic
    models its parts already returned, so ``at()`` costs one linearization of
    the forward model however the objective is assembled."""

    @staticmethod
    def misfit(space, counts):
        """``0.5 |F(x)|^2`` with ``F`` the componentwise square, built from
        one ``linearize`` callable that also gives a Hessian."""
        from pygeoinf.algebra.linearization import QuadraticModel

        def value(x):
            return 0.5 * float(np.sum(space.to_components(x) ** 4))

        def linearize(x):
            counts["linearize"] += 1
            c = space.to_components(x)
            derivative = LinearFunctional.from_derivative_components(space, 2.0 * c**3)
            hessian = LinearOperator.from_matrix(
                space, space, np.diag(6.0 * c**2), form="components"
            )
            return QuadraticModel(x, value(x), derivative, hessian)

        def hessian(x):
            counts["hessian"] += 1
            return linearize(x).hessian

        return Functional.from_callables(
            space, value, linearize=linearize, hessian=hessian
        )

    def test_sums_and_scalings_reuse_the_models(self, rng):
        X = make_weighted_space()
        counts = {"linearize": 0, "hessian": 0}
        phi = self.misfit(X, counts)
        regularizer = quadratic_on(X, np.identity(X.dim))
        x = X.random(rng=rng)
        for objective in (
            phi,
            0.5 * phi,
            phi + regularizer,
            0.5 * phi + regularizer,
            -0.5 * (phi + regularizer),
        ):
            counts.update(linearize=0, hessian=0)
            model = objective.at(x)
            assert model.hessian is not None
            assert counts == {"linearize": 1, "hessian": 0}, objective
            dx = X.random(rng=rng)
            assert np.allclose(
                X.to_components(model.hessian(dx)),
                X.to_components(objective.hessian(x)(dx)),
            )


class TestALinearizationAndAHessianSuppliedApart:
    def test_the_model_carries_the_hessian(self, rng):
        """``linearize=`` without a Hessian beside ``hessian=``: ``at()``
        returns a model with the Hessian, as ``has_hessian`` promises."""
        from pygeoinf.algebra.linearization import QuadraticModel

        X = make_dense_metric_space(3)

        def value(x):
            return 0.5 * X.squared_norm(x)

        def linearize(x):
            return QuadraticModel(x, value(x), LinearFunctional.from_representer(X, x))

        f = Functional.from_callables(
            X, value, linearize=linearize, hessian=lambda x: LinearOperator.identity(X)
        )
        assert f.has_hessian
        x = X.random(rng=rng)
        model = f.at(x)
        assert model.hessian is not None
        from pygeoinf.numerics.optimization import NewtonCG

        result = NewtonCG().minimize(f, x)
        assert X.norm(result.minimizer) < 1e-6


class TestCheckHessian:
    """``check_hessian``: the Hessian against differences of the gradient,
    on a dense metric, where a Hessian written in components rather than
    lifted by the inverse metric is wrong by a factor of the Gram matrix."""

    def test_a_correct_hessian_passes(self, rng):
        from pygeoinf.testing import check_hessian

        X = make_dense_metric_space(4)
        M = rng.normal(size=(4, 4))
        phi = quadratic_on(X, M @ M.T + np.identity(4))
        report = check_hessian(phi, X.random(rng=rng), rng=rng)
        assert (
            report.defects["the Hessian agrees with differences of the gradient"] < 1e-6
        )
        assert "the Hessian is self-adjoint" in report.defects

    def test_a_composed_hessian_passes(self, rng):
        from pygeoinf.testing import check_hessian

        X = make_dense_metric_space(3)
        Y = make_weighted_space()
        A = LinearOperator.from_matrix(X, Y, rng.normal(size=(4, 3)), form="components")
        phi = quadratic_on(Y, np.identity(4))
        check_hessian(0.5 * (phi @ A) + phi @ A, X.random(rng=rng), rng=rng)

    def test_a_hessian_missing_the_inverse_metric_fails(self, rng):
        from pygeoinf.testing import check_hessian

        X = make_dense_metric_space(4)
        M = np.identity(4) * 2.0
        good = quadratic_on(X, M)
        wrong = Functional.from_callables(
            X,
            good,
            derivative=good.derivative,
            hessian=lambda x: LinearOperator.self_adjoint(
                X, lambda v: X.from_components(M @ X.to_components(v))
            ),
        )
        with pytest.raises(AssertionError, match="differences of the gradient"):
            check_hessian(wrong, X.random(rng=rng), rng=rng)

    def test_a_functional_without_one_is_refused(self, rng):
        from pygeoinf.testing import check_hessian

        X = EuclideanSpace(3)
        f = Functional.from_callables(
            X, lambda x: float(x @ x), gradient=lambda x: 2.0 * x
        )
        with pytest.raises(AssertionError, match="carries a Hessian"):
            check_hessian(f, X.random(rng=rng), rng=rng)


class TestAnAffineMapInTheSecondOrderAlgebra:
    """An affine map's second derivative is zero, so ``F @ (A + b)`` keeps
    the second derivative ``F @ A`` has, ``phi @ (A + b)`` keeps a Hessian,
    and a linear functional composed with one is a functional."""

    @staticmethod
    def setting(rng):
        X, Y = make_dense_metric_space(3), make_weighted_space()
        A = LinearOperator.from_matrix(X, Y, rng.normal(size=(4, 3)), form="components")
        return X, Y, AffineOperator(A, Y.random(rng=rng))

    def test_the_second_derivative_is_zero(self, rng):
        from pygeoinf.testing import check_hessian

        X, Y, F = self.setting(rng)
        assert F.has_second_derivative
        x = X.random(rng=rng)
        assert Y.norm(F.second_derivative(x, X.random(rng=rng))(x)) == 0.0
        phi = quadratic_on(Y, np.identity(Y.dim))
        assert (phi @ F).has_hessian
        check_hessian(phi @ F, x, rng=rng)

    def test_a_linear_functional_after_it_is_a_functional(self, rng):
        from pygeoinf.testing import check_gradient

        X, Y, F = self.setting(rng)
        functional = LinearFunctional.from_representer(Y, Y.random(rng=rng))
        composed = functional @ F
        assert isinstance(composed, Functional)
        x = X.random(rng=rng)
        assert composed(x) == pytest.approx(functional(F(x)))
        check_gradient(composed, x, rng=rng)
