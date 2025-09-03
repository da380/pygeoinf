"""
Module containing an abstract test class for NonLinearOperator implementations.
"""

from __future__ import annotations
import pytest
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pygeoinf.hilbert_space import HilbertSpace, Vector
    from pygeoinf.nonlinear_operators import NonLinearOperator


class NonLinearOperatorChecks:
    """
    An abstract base class for testing NonLinearOperator implementations.

    To use this, inherit from it and provide fixtures named `operator` and
    `operator2` that return NonLinearOperator instances for testing.
    """

    # --- Fixtures ---
    @pytest.fixture
    def domain(self, operator: "NonLinearOperator") -> "HilbertSpace":
        return operator.domain

    @pytest.fixture
    def codomain(self, operator: "NonLinearOperator") -> "HilbertSpace":
        return operator.codomain

    @pytest.fixture
    def x(self, domain: "HilbertSpace") -> Vector:
        """A random vector from the operator's domain."""
        return domain.random()

    # --- Basic Sanity Checks ---

    def test_output_codomain(self, operator: "NonLinearOperator", x: Vector):
        """Tests that the operator's output is compatible with the codomain."""
        Fx = operator(x)
        try:
            _ = operator.codomain.add(Fx, operator.codomain.zero)
        except Exception as e:
            pytest.fail(f"Operator output is not compatible with codomain: {e}")

    # --- Algebraic Consistency Checks ---

    def test_scalar_multiplication_consistency(
        self, operator: "NonLinearOperator", x: Vector
    ):
        """Tests that (a * F)(x) is consistent with a * F(x)."""
        scalar = 2.5
        scaled_op = scalar * operator

        # Calculate lhs: a * F(x)
        Fx = operator(x)
        lhs_components = operator.codomain.to_components(
            operator.codomain.multiply(scalar, Fx)
        )

        # Calculate rhs: (a * F)(x)
        rhs_components = operator.codomain.to_components(scaled_op(x))

        assert np.allclose(lhs_components, rhs_components)

    def test_addition_consistency(
        self, operator: "NonLinearOperator", operator2: "NonLinearOperator", x: Vector
    ):
        """Tests that (F + G)(x) is consistent with F(x) + G(x)."""
        sum_op = operator + operator2

        # Calculate lhs: F(x) + G(x)
        Fx = operator(x)
        Gx = operator2(x)
        lhs_components = operator.codomain.to_components(operator.codomain.add(Fx, Gx))

        # Calculate rhs: (F + G)(x)
        rhs_components = operator.codomain.to_components(sum_op(x))

        assert np.allclose(lhs_components, rhs_components)

    def test_composition_consistency(
        self, operator: "NonLinearOperator", operator2: "NonLinearOperator", x: Vector
    ):
        """Tests that (F @ G)(x) is consistent with F(G(x))."""
        # Ensure operators can be composed
        if operator.domain != operator2.codomain:
            pytest.skip("Cannot compose operators: codomain/domain mismatch.")

        composed_op = operator @ operator2

        # Calculate lhs: F(G(x))
        Gx = operator2(x)
        lhs_components = operator.codomain.to_components(operator(Gx))

        # Calculate rhs: (F @ G)(x)
        rhs_components = composed_op.codomain.to_components(composed_op(x))

        assert np.allclose(lhs_components, rhs_components)
