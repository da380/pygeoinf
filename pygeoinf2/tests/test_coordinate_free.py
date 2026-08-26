"""
The core must work with no coordinate map at all.

Every other test in the suite uses NumPy-backed spaces whose vectors *are*
their components, so code reaching for array arithmetic or a component map
works by accident. These tests remove the accident, using vectors that support
no arithmetic and spaces that raise if their coordinate map is touched.

This is what stands in for a real foreign backend, without needing one installed.
"""

import numpy as np
import pytest

from pygeoinf2.algebra.operators import Functional, LinearFunctional, LinearOperator
from pygeoinf2.testing import (
    check_gradient,
    check_operator,
    check_space,
    check_traits,
    check_white_noise,
)
from pygeoinf2.traits import Traits

from .conftest import make_weighted_space
from .doubles import NoCoordinatesError, Opaque, OpaqueSpace, StrictSpace


WEIGHTS = np.array([1.0, 4.0, 9.0, 0.25])


def make_opaque_space():
    return OpaqueSpace(WEIGHTS)


class TestOpaqueVectors:
    def test_the_double_really_refuses_arithmetic(self):
        """If this passes trivially, the rest of the file proves nothing."""
        a, b = Opaque(np.ones(3)), Opaque(np.ones(3))
        with pytest.raises(TypeError):
            a + b
        with pytest.raises(TypeError):
            2.0 * a
        assert not hasattr(a, "copy")

    def test_space_axioms_hold(self, rng):
        check_space(make_opaque_space(), rng=rng, rebuild=make_opaque_space)

    def test_white_noise_is_white(self, rng):
        check_white_noise(make_opaque_space(), rng=rng, samples=40000, rtol=0.05)

    def test_it_is_not_a_coordinate_space(self):
        from pygeoinf2.algebra.spaces import CoordinateSpace

        assert not isinstance(make_opaque_space(), CoordinateSpace)


class TestOperatorsWithoutCoordinates:
    def test_adjoint_identity(self, rng):
        X = make_opaque_space()
        weights = np.array([2.0, -1.0, 0.5, 3.0])

        A = LinearOperator.from_callables(
            X,
            X,
            lambda x: Opaque(weights * x.data),
            adjoint=lambda y: Opaque(weights * y.data),
            traits=Traits.SELF_ADJOINT,
        )
        check_operator(A, rng=rng)
        check_traits(A, rng=rng)

    def test_algebra_and_trait_propagation(self, rng):
        X = make_opaque_space()
        Y = OpaqueSpace(np.array([1.0, 2.0]))
        matrix = np.array([[1.0, 0.5, -0.5, 2.0], [0.0, 1.0, 1.0, -1.0]])

        def forward(x):
            return Opaque(matrix @ x.data)

        def backward(y):
            # A* = G_X^-1 M^T G_Y, written out by hand as a backend would.
            return Opaque((matrix.T @ (Y._weights * y.data)) / X._weights)

        A = LinearOperator.from_callables(X, Y, forward, adjoint=backward)
        check_operator(A, rng=rng)

        gramian = A @ A.adjoint
        assert Traits.POSITIVE_SEMIDEFINITE & gramian.traits
        check_operator(gramian, rng=rng)
        check_traits(gramian, rng=rng)

    def test_functionals_work_without_components(self, rng):
        X = make_opaque_space()
        v = X.random(rng=rng)
        f = LinearFunctional.from_representer(X, v)
        x = X.random(rng=rng)
        assert f(x) == pytest.approx(X.inner_product(v, x))
        assert X.inner_product(f.representer, x) == pytest.approx(f(x))

    def test_gradient_check_without_components(self, rng):
        X = make_opaque_space()
        centre = X.random(rng=rng)

        def value(x):
            d = X.subtract(x, centre)
            return 0.5 * X.squared_norm(d)

        def gradient(x):
            return X.subtract(x, centre)

        phi = Functional.from_callables(X, value, gradient=gradient)
        check_gradient(phi, X.random(rng=rng), rng=rng)

    def test_matrix_is_refused(self, rng):
        X = make_opaque_space()
        A = LinearOperator.self_adjoint(X, lambda x: x)
        with pytest.raises(TypeError, match="no coordinate map"):
            A.matrix()


class TestStrictSpaceProvesCoordinateFreedom:
    """A space that raises when its coordinate map is touched.

    This turns "coordinate-free" from a claim into an assertion: if any of
    these paths reaches for components, the test fails loudly.
    """

    def test_the_guard_actually_fires(self):
        strict = StrictSpace(make_weighted_space())
        with pytest.raises(NoCoordinatesError):
            strict.to_components(strict.zero())
        with pytest.raises(NoCoordinatesError):
            strict.from_components(np.zeros(4))

    def test_space_axioms_are_coordinate_free(self, rng):
        check_space(StrictSpace(make_weighted_space()), rng=rng)

    def test_operator_and_trait_checks_are_coordinate_free(self, rng):
        base = make_weighted_space()
        strict = StrictSpace(base)

        def scale(x):
            return base.scale(3.0, x)

        A = LinearOperator.self_adjoint(strict, scale)
        check_operator(A, rng=rng)
        check_traits(A, rng=rng)

    def test_the_algebra_is_coordinate_free(self, rng):
        base = make_weighted_space()
        strict = StrictSpace(base)
        A = LinearOperator.self_adjoint(strict, lambda x: base.scale(3.0, x))
        B = LinearOperator.self_adjoint(strict, lambda x: base.scale(-1.0, x))

        combined = (2.0 * A + B) @ A
        check_operator(combined, rng=rng)
        assert Traits.POSITIVE_SEMIDEFINITE & (A @ A.adjoint).traits

    def test_gradient_checking_is_coordinate_free(self, rng):
        base = make_weighted_space()
        strict = StrictSpace(base)
        centre = strict.random(rng=rng)

        phi = Functional.from_callables(
            strict,
            lambda x: 0.5 * strict.squared_norm(strict.subtract(x, centre)),
            gradient=lambda x: strict.subtract(x, centre),
        )
        check_gradient(phi, strict.random(rng=rng), rng=rng)

    def test_a_coordinate_using_path_is_caught(self, rng):
        """The negative control: matrix() does need coordinates, and says so."""
        strict = StrictSpace(make_weighted_space())
        A = LinearOperator.self_adjoint(strict, lambda x: x)
        with pytest.raises(NoCoordinatesError):
            A.matrix(form="components")
