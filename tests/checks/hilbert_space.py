"""
Module containing an abstract test class for Hilbert space implementations.

This class defines a "contract" of tests that any concrete implementation of
the `HilbertSpace` class must pass to be considered valid.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import pytest
import numpy as np


from numpy.testing import assert_allclose

# This block only runs for type checkers, not at runtime
if TYPE_CHECKING:
    from pygeoinf.hilbert_space import HilbertSpace, T_vec


class HilbertSpaceChecks:
    """
    An abstract base class for testing Hilbert space implementations.

    To use this, create a concrete test class that inherits from this one
    and provide a pytest fixture named `space` that returns an instance of
    the HilbertSpace subclass you want to test.
    """

    # =========================================================================
    # Pytest Fixtures
    # These fixtures provide the necessary objects for the tests.
    # =========================================================================

    @pytest.fixture
    def x(self, space: "HilbertSpace") -> "T_vec":
        """A random vector from the Hilbert space."""
        return space.random()

    @pytest.fixture
    def y(self, space: "HilbertSpace") -> "T_vec":
        """A second random vector from the Hilbert space."""
        return space.random()

    @pytest.fixture
    def z(self, space: "HilbertSpace") -> "T_vec":
        """A third random vector from the Hilbert space."""
        return space.random()

    @pytest.fixture
    def a(self) -> float:
        """A random scalar."""
        return np.random.randn()

    @pytest.fixture
    def b(self) -> float:
        """A second random scalar."""
        return np.random.randn()

    # =========================================================================
    # Vector Space Axiom Tests
    # =========================================================================

    def test_add_subtract_identity(self, space: "HilbertSpace", x: "T_vec", y: "T_vec"):
        """Tests if (x + y) - y == x."""
        sum_vec = space.add(x, y)
        result_vec = space.subtract(sum_vec, y)
        # Compare the component representations for numerical equality
        assert np.allclose(space.to_components(x), space.to_components(result_vec))

    def test_multiply_divide_identity(
        self, space: "HilbertSpace", x: "T_vec", a: float
    ):
        """Tests if (a * x) / a == x for a non-zero scalar."""
        if np.isclose(a, 0):
            pytest.skip("Scalar is too close to zero for division test.")
        scaled_vec = space.multiply(a, x)
        result_vec = space.multiply(1.0 / a, scaled_vec)
        assert np.allclose(space.to_components(x), space.to_components(result_vec))

    def test_ax(self, space: "HilbertSpace", x: "T_vec", a: float):
        """Tests the in-place operation x := a*x."""
        x_copy = space.copy(x)
        # Expected result computed with out-of-place operation
        expected_result = space.multiply(a, x)
        # Perform the in-place operation
        space.ax(a, x_copy)
        assert np.allclose(
            space.to_components(expected_result), space.to_components(x_copy)
        )

    def test_axpy(self, space: "HilbertSpace", x: "T_vec", y: "T_vec", a: float):
        """Tests the in-place operation y := a*x + y."""
        y_copy = space.copy(y)
        # Expected result computed with out-of-place operations
        expected_result = space.add(space.multiply(a, x), y)
        # Perform the in-place operation
        space.axpy(a, x, y_copy)
        assert np.allclose(
            space.to_components(expected_result), space.to_components(y_copy)
        )

    def test_distributivity(
        self, space: "HilbertSpace", x: "T_vec", y: "T_vec", a: float
    ):
        """Tests the distributive property: a*(x+y) == a*x + a*y."""
        lhs = space.multiply(a, space.add(x, y))
        rhs = space.add(space.multiply(a, x), space.multiply(a, y))
        assert np.allclose(space.to_components(lhs), space.to_components(rhs))

    def test_zero_vector_add_identity(self, space: "HilbertSpace", x: "T_vec"):
        """Tests the additive identity property: x + 0 = x."""
        zero_vec = space.zero
        result_vec = space.add(x, zero_vec)
        assert np.allclose(space.to_components(x), space.to_components(result_vec))

    def test_zero_vector_multiplication(self, space: "HilbertSpace", x: "T_vec"):
        """Tests that 0 * x = 0."""
        zero_vec = space.zero
        result_vec = space.multiply(0.0, x)
        assert np.allclose(
            space.to_components(zero_vec), space.to_components(result_vec)
        )

    # =========================================================================
    # Inner Product and Norm Axiom Tests
    # =========================================================================

    def test_inner_product_linearity(
        self,
        space: "HilbertSpace",
        x: "T_vec",
        y: "T_vec",
        z: "T_vec",
        a: float,
        b: float,
    ):
        """Tests linearity of the inner product in the first argument."""
        lhs = space.inner_product(
            space.add(space.multiply(a, x), space.multiply(b, y)), z
        )
        rhs = a * space.inner_product(x, z) + b * space.inner_product(y, z)
        assert np.isclose(lhs, rhs)

    def test_inner_product_symmetry(
        self, space: "HilbertSpace", x: "T_vec", y: "T_vec"
    ):
        """Tests symmetry of the inner product: <x, y> == <y, x>."""
        ip1 = space.inner_product(x, y)
        ip2 = space.inner_product(y, x)
        assert np.isclose(ip1, ip2)

    def test_inner_product_positive_definite(self, space: "HilbertSpace", x: "T_vec"):
        """Tests positive-definiteness: <x, x> >= 0 and <x, x> = 0 iff x = 0."""
        norm_sq = space.squared_norm(x)
        assert norm_sq >= 0
        # Check that the norm of the zero vector is zero
        zero_norm_sq = space.squared_norm(space.zero)
        assert np.isclose(zero_norm_sq, 0)

    def test_norm_triangle_inequality(
        self, space: "HilbertSpace", x: "T_vec", y: "T_vec"
    ):
        """Tests the triangle inequality: ||x + y|| <= ||x|| + ||y||."""
        norm_x = space.norm(x)
        norm_y = space.norm(y)
        norm_sum = space.norm(space.add(x, y))
        assert norm_sum <= norm_x + norm_y

    def test_norm_scalar_multiplication(
        self, space: "HilbertSpace", x: "T_vec", a: float
    ):
        """Tests scalar multiplication property of the norm: ||a*x|| = |a|*||x||."""
        norm_ax = space.norm(space.multiply(a, x))
        abs_a_norm_x = np.abs(a) * space.norm(x)
        assert np.isclose(norm_ax, abs_a_norm_x)

    # =========================================================================
    # Component and Dual Mapping Tests
    # =========================================================================

    def test_to_from_components_identity(self, space: "HilbertSpace", x: "T_vec"):
        """Ensures that from_components(to_components(x)) is the identity."""
        components = space.to_components(x)
        reconstructed_x = space.from_components(components)
        assert np.allclose(components, space.to_components(reconstructed_x))

    def test_riesz_representation_theorem(
        self, space: "HilbertSpace", x: "T_vec", y: "T_vec"
    ):
        """
        Tests the Riesz representation theorem: <x, y> = (to_dual(y))(x).

        This is a fundamental check that the inner product and the mapping to the
        dual space are consistent.
        """

        inner_product_val = space.inner_product(x, y)
        yp = space.to_dual(y)
        linear_form_val = space.duality_product(yp, x)
        assert np.isclose(inner_product_val, linear_form_val)

    def test_to_from_dual_identity(self, space: "HilbertSpace", x: "T_vec"):
        """Ensures that from_dual(to_dual(x)) is the identity."""
        x_dual = space.to_dual(x)
        reconstructed_x = space.from_dual(x_dual)
        assert np.allclose(space.to_components(x), space.to_components(reconstructed_x))
