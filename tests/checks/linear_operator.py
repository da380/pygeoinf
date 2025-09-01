"""
Module containing an abstract test class for LinearOperator implementations.

This class defines a "contract" of tests that any concrete implementation of
the `LinearOperator` class should pass.
"""

from __future__ import annotations
import pytest
import numpy as np
from typing import TYPE_CHECKING

# This block only runs for type checkers, not at runtime
if TYPE_CHECKING:
    from pygeoinf.hilbert_space import HilbertSpace, Vector
    from pygeoinf.operators import LinearOperator
    from pygeoinf.linear_forms import LinearForm


class LinearOperatorChecks:
    """
    An abstract base class for testing LinearOperator implementations.

    To use this, create a concrete test class that inherits from this one
    and provide a pytest fixture named `operator` that returns an instance of
    the LinearOperator you want to test.
    """

    # =========================================================================
    # Pytest Fixtures
    # =========================================================================

    @pytest.fixture
    def domain(self, operator: "LinearOperator") -> "HilbertSpace":
        """Provides the domain of the operator."""
        return operator.domain

    @pytest.fixture
    def codomain(self, operator: "LinearOperator") -> "HilbertSpace":
        """Provides the codomain of the operator."""
        return operator.codomain

    @pytest.fixture
    def x(self, domain: "HilbertSpace") -> Vector:
        """A random vector from the operator's domain."""
        return domain.random()

    @pytest.fixture
    def x2(self, domain: "HilbertSpace") -> Vector:
        """A second random vector from the operator's domain for linearity tests."""
        return domain.random()

    @pytest.fixture
    def y(self, codomain: "HilbertSpace") -> Vector:
        """A random vector from the operator's codomain for adjoint tests."""
        return codomain.random()

    @pytest.fixture
    def yp(self, codomain: "HilbertSpace", y: Vector) -> "LinearForm":
        """A random dual vector from the codomain's dual space."""
        return codomain.to_dual(y)

    @pytest.fixture
    def a(self) -> float:
        """A random scalar."""
        return np.random.randn()

    @pytest.fixture
    def b(self) -> float:
        """A second random scalar."""
        return np.random.randn()

    # =========================================================================
    # Core Operator Property Tests
    # =========================================================================

    def test_adjoint(
        self,
        operator: "LinearOperator",
        domain: "HilbertSpace",
        codomain: "HilbertSpace",
        x: Vector,
        y: Vector,
    ):
        """
        Tests the adjoint property: <A(x), y>_codomain = <x, A*(y)>_domain.

        This is the most fundamental test for a linear operator in a Hilbert
        space framework. Its success is critical for the correctness of
        many inversion algorithms.
        """
        # Calculate the left-hand side of the identity
        Ax = operator(x)
        lhs = codomain.inner_product(Ax, y)

        # Calculate the right-hand side of the identity
        A_star_y = operator.adjoint(y)
        rhs = domain.inner_product(x, A_star_y)

        # Check for numerical equality
        assert np.isclose(lhs, rhs)

    def test_dual_definition(
        self,
        operator: "LinearOperator",
        x: Vector,
        yp: "LinearForm",
    ):
        """
        Tests the defining property of the dual operator A': (A'(yp))(x) = yp(A(x)).
        """
        # Calculate the left-hand side: apply the dual operator to a dual vector,
        # then apply the resulting dual vector (a LinearForm) to a primal vector.
        A_prime_yp = operator.dual(yp)
        lhs = A_prime_yp(x)

        # Calculate the right-hand side: apply the primal operator to a primal
        # vector, then apply the original dual vector to the result.
        Ax = operator(x)
        rhs = yp(Ax)

        assert np.isclose(lhs, rhs)

    def test_linearity(
        self,
        operator: "LinearOperator",
        domain: "HilbertSpace",
        x: Vector,
        x2: Vector,
        a: float,
        b: float,
    ):
        """
        Tests the linearity property: A(a*x + b*x2) = a*A(x) + b*A(x2).
        """
        # Calculate the left-hand side
        x_comb = domain.add(domain.multiply(a, x), domain.multiply(b, x2))
        lhs = operator(x_comb)

        # Calculate the right-hand side
        Ax = operator(x)
        Ax2 = operator(x2)
        rhs = operator.codomain.add(
            operator.codomain.multiply(a, Ax), operator.codomain.multiply(b, Ax2)
        )

        # Compare the component representations for numerical equality
        assert np.allclose(
            operator.codomain.to_components(lhs),
            operator.codomain.to_components(rhs),
            atol=1e-14,
        )
