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
        x = X.random(rng)

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
            check_gradient(phi, X.random(rng), rng=rng)

    def test_the_error_is_invisible_on_an_orthonormal_space(self, rng):
        """Which is why it survives: the toy case cannot detect it."""
        X = EuclideanSpace(4)
        M = 0.5 * (lambda A: A + A.T)(rng.normal(size=(4, 4)))

        def value(x):
            return 0.5 * float(x @ M @ x)

        phi = Functional.from_callables(X, value, gradient=lambda x: M @ x)
        check_gradient(phi, X.random(rng), rng=rng)  # passes, correctly

    def test_at_returns_a_quadratic_model(self, rng):
        X = make_weighted_space()
        phi = quadratic_on(X, rng.normal(size=(X.dim, X.dim)))
        model = phi.at(X.random(rng))
        assert isinstance(model, QuadraticModel)
        assert model.has_hessian
        assert isinstance(model.value, float)

    def test_gradient_is_cached(self, rng):
        X = make_weighted_space()
        phi = quadratic_on(X, rng.normal(size=(X.dim, X.dim)))
        model = phi.at(X.random(rng))
        assert model.gradient is model.gradient

    def test_hessian_is_absent_unless_supplied(self, rng):
        X = EuclideanSpace(3)
        phi = Functional.from_callables(
            X, lambda x: float(x @ x), gradient=lambda x: 2.0 * x
        )
        assert not phi.has_hessian
        assert phi.at(X.random(rng)).hessian is None

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
        x = X.random(rng)
        assert f(x) == pytest.approx(float(g @ X.to_components(x)))

    def test_from_representer_pairs_by_inner_product(self, rng):
        X = make_weighted_space()
        v = X.random(rng)
        f = LinearFunctional.from_representer(X, v)
        x = X.random(rng)
        assert f(x) == pytest.approx(X.inner_product(v, x))

    def test_works_without_coordinates(self, rng):
        """from_representer needs no component map at all."""
        X = OpaqueSpace(np.array([1.0, 2.0, 3.0]))
        v = X.random(rng)
        f = LinearFunctional.from_representer(X, v)
        x = X.random(rng)
        assert f(x) == pytest.approx(X.inner_product(v, x))

    def test_hessian_is_zero(self, rng):
        X = EuclideanSpace(3)
        f = LinearFunctional.from_derivative_components(X, np.array([1.0, 2.0, 3.0]))
        assert np.allclose(f.hessian(X.random(rng))(X.random(rng)), np.zeros(3))


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
            return LinearOperator.from_component_matrix(
                X, Y, np.stack([2.0 * (B @ c) for B in blocks])
            )

        def second_derivative(x, dx):
            cd = X.to_components(dx)
            return LinearOperator.from_component_matrix(
                X, Y, np.stack([2.0 * (B @ cd) for B in blocks])
            )

        return Operator.from_callables(
            X, Y, value, derivative=derivative, second_derivative=second_derivative
        )

    def test_derivative(self, rng):
        X, Y = make_weighted_space(), EuclideanSpace(2)
        F = self.build(X, Y, rng)
        check_derivative(F, X.random(rng), rng=rng)

    def test_second_derivative(self, rng):
        X, Y = make_weighted_space(), EuclideanSpace(2)
        F = self.build(X, Y, rng)
        check_second_derivative(F, X.random(rng), rng=rng)

    def test_chain_rule(self, rng):
        X, Y, Z = make_weighted_space(), EuclideanSpace(3), EuclideanSpace(2)
        F = self.build(X, Y, rng)
        A = LinearOperator.from_component_matrix(Y, Z, rng.normal(size=(2, 3)))
        composed = A @ F
        assert composed.has_derivative
        check_derivative(composed, X.random(rng), rng=rng)

    def test_second_derivative_survives_composition(self, rng):
        """(F o G)'' needs both factors to carry one."""
        X, Y, Z = make_weighted_space(), EuclideanSpace(3), EuclideanSpace(2)
        F = self.build(X, Y, rng)
        G = self.build(Y, Z, rng)
        composed = G @ F
        assert composed.has_second_derivative
        check_second_derivative(composed, X.random(rng), rng=rng, rtol=1e-3)

    def test_second_derivative_absent_when_a_factor_lacks_one(self, rng):
        X, Y, Z = make_weighted_space(), EuclideanSpace(3), EuclideanSpace(2)
        F = self.build(X, Y, rng)
        partial = Operator.from_callables(
            Y,
            Z,
            lambda y: np.array([y @ y, 0.0]),
            derivative=lambda y: LinearOperator.from_component_matrix(
                Y, Z, np.stack([2.0 * y, np.zeros(3)])
            ),
        )
        assert not (partial @ F).has_second_derivative

    def test_sum_and_scaling(self, rng):
        X, Y = make_weighted_space(), EuclideanSpace(2)
        F = self.build(X, Y, rng)
        G = self.build(X, Y, rng)
        for op in (F + G, 2.5 * F, F - G):
            check_derivative(op, X.random(rng), rng=rng)


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
                x, value, LinearOperator.from_component_matrix(X, Y, M)
            )

        F = Operator.from_callables(X, Y, lambda x: M @ x, linearise=linearise)

        model = F.at(X.random(rng))
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
                LinearOperator.from_component_matrix(X, Y, M),
            )

        F = Operator.from_callables(X, Y, lambda x: M @ x, linearise=linearise)
        for _ in range(20):
            F(X.random(rng))
        assert calls["n"] == 0


class TestAffineOperator:
    def test_value_and_derivative(self, rng):
        X, Y = make_weighted_space(), EuclideanSpace(3)
        A = LinearOperator.from_component_matrix(X, Y, rng.normal(size=(3, X.dim)))
        b = Y.random(rng)
        F = AffineOperator(A, b)
        x = X.random(rng)
        assert np.allclose(F(x), A(x) + b)
        assert F.derivative(x) is A
        check_derivative(F, x, rng=rng)

    def test_affineness_survives_the_algebra(self, rng):
        """v1 preserves this with a string type check; the protocol replaces it."""
        X, Y = EuclideanSpace(4), EuclideanSpace(3)
        A = LinearOperator.from_component_matrix(X, Y, rng.normal(size=(3, 4)))
        B = LinearOperator.from_component_matrix(X, Y, rng.normal(size=(3, 4)))
        F = AffineOperator(A, Y.random(rng))

        assert isinstance(F + B, AffineOperator)
        assert isinstance(B + F, AffineOperator)  # the order-independent case
        assert isinstance(2.0 * F, AffineOperator)
        assert isinstance(F + F, AffineOperator)

    def test_composition_with_linear_operators_stays_affine(self, rng):
        X, Y, Z = EuclideanSpace(4), EuclideanSpace(3), EuclideanSpace(2)
        A = LinearOperator.from_component_matrix(X, Y, rng.normal(size=(3, 4)))
        C = LinearOperator.from_component_matrix(Y, Z, rng.normal(size=(2, 3)))
        D = LinearOperator.from_component_matrix(Z, X, rng.normal(size=(4, 2)))
        F = AffineOperator(A, Y.random(rng))

        assert isinstance(C @ F, AffineOperator)
        assert isinstance(F @ D, AffineOperator)
        x = Z.random(rng)
        assert np.allclose((C @ F)(D(x)), C(F(D(x))))

    def test_linearisation_as_affine(self, rng):
        X, Y = make_weighted_space(), EuclideanSpace(2)
        F = TestNonlinearOperator().build(X, Y, rng)
        x = X.random(rng)
        model = F.at(x)
        affine = model.as_affine()
        assert np.allclose(affine(x), model.value)
