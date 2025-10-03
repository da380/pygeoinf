"""
Provides the `NonLinearOperator` base class for mappings between Hilbert spaces.

A non-linear operator is a general mapping `F(x)` from a vector `x` in a
domain Hilbert space to a vector `y` in a codomain Hilbert space. This class
provides a foundational structure for these mappings, equipping them with
algebraic operations and an interface for the Frécher derivative.
"""

from __future__ import annotations
from typing import Callable, Any, TYPE_CHECKING


from .checks.nonlinear_operators import NonLinearOperatorAxiomChecks


# This block only runs for type checkers, not at runtime
if TYPE_CHECKING:
    from .hilbert_space import HilbertSpace, Vector
    from .linear_operators import LinearOperator


class NonLinearOperator(NonLinearOperatorAxiomChecks):
    """
    Represents a general non-linear operator that maps vectors to vectors.

    This class provides a functional representation for an operator `F(x)`,
    and includes an interface for its Fréchet derivative, F'(x), which is the
    linear operator that best approximates F at a given point x. It serves
    as the base class for the more specialized `LinearOperator`.
    """

    def __init__(
        self,
        domain: HilbertSpace,
        codomain: HilbertSpace,
        mapping: Callable[[Vector], Any],
        /,
        *,
        derivative: Callable[[Vector], LinearOperator] = None,
    ) -> None:
        """Initializes the NonLinearOperator.

        Args:
            domain: The Hilbert space from which the operator maps.
            codomain: The Hilbert space to which the operator maps.
            mapping: The function `F(x)` that defines the mapping.
            derivative: An optional function that takes a vector `x` and
                returns the Fréchet derivative (a `LinearOperator`) at
                that point.
        """
        self._domain: HilbertSpace = domain
        self._codomain: HilbertSpace = codomain
        self._mapping: Callable[[Any], Any] = mapping
        self._derivative: Callable[[Any], LinearOperator] = derivative

    @property
    def domain(self) -> HilbertSpace:
        """The domain of the operator."""
        return self._domain

    @property
    def codomain(self) -> HilbertSpace:
        """The codomain of the operator."""
        return self._codomain

    @property
    def is_automorphism(self) -> bool:
        """True if the operator maps a space into itself."""
        return self.domain == self.codomain

    @property
    def is_square(self) -> bool:
        """True if the operator's domain and codomain have the same dimension."""
        return self.domain.dim == self.codomain.dim

    @property
    def has_derivative(self) -> bool:
        """
        Returns true if the operators derivative is implemented.
        """
        return self._derivative is not None

    def __call__(self, x: Any) -> Any:
        """Applies the operator's mapping to a vector."""
        return self._mapping(x)

    def derivative(self, x: Vector) -> LinearOperator:
        """Computes the Fréchet derivative of the operator at a given point.

        The Fréchet derivative is the linear operator that best approximates
        the non-linear operator in the neighborhood of the point `x`.

        Args:
            x: The point at which to compute the derivative.

        Returns:
            The derivative as a `LinearOperator`.

        Raises:
            NotImplementedError: If a derivative function was not provided.
        """
        if self._derivative is None:
            raise NotImplementedError("Derivative not implemented")
        return self._derivative(x)

    def __neg__(self) -> NonLinearOperator:
        domain = self.domain
        codomain = self.codomain

        def mapping(x: Any) -> Any:
            return codomain.negative(self(x))

        if self._derivative is not None:

            def derivative(x: Vector) -> LinearOperator:
                return -self.derivative(x)

        else:
            derivative = None

        return NonLinearOperator(domain, codomain, mapping, derivative=derivative)

    def __mul__(self, a: float) -> NonLinearOperator:
        domain = self.domain
        codomain = self.codomain

        def mapping(x: Any) -> Any:
            return codomain.multiply(a, self(x))

        if self._derivative is not None:

            def derivative(x: Vector) -> LinearOperator:
                return a * self.derivative(x)

        else:
            derivative = None

        return NonLinearOperator(domain, codomain, mapping, derivative=derivative)

    def __rmul__(self, a: float) -> NonLinearOperator:
        return self * a

    def __truediv__(self, a: float) -> NonLinearOperator:
        return self * (1.0 / a)

    def __add__(self, other: NonLinearOperator) -> NonLinearOperator:

        if not isinstance(other, NonLinearOperator):
            raise TypeError("Operand must be a NonLinearOperator")

        domain = self.domain
        codomain = self.codomain

        def mapping(x: Any) -> Any:
            return codomain.add(self(x), other(x))

        if self._derivative is not None and other._derivative is not None:

            def derivative(x: Vector) -> LinearOperator:
                return self.derivative(x) + other.derivative(x)

        else:
            derivative = None

        return NonLinearOperator(domain, codomain, mapping, derivative=derivative)

    def __sub__(self, other: NonLinearOperator) -> NonLinearOperator:

        if not isinstance(other, NonLinearOperator):
            raise TypeError("Operand must be a NonLinearOperator")

        domain = self.domain
        codomain = self.codomain

        def mapping(x: Any) -> Any:
            return codomain.subtract(self(x), other(x))

        if self._derivative is not None and other._derivative is not None:

            def derivative(x: Vector) -> LinearOperator:
                return self.derivative(x) - other.derivative(x)

        else:
            derivative = None

        return NonLinearOperator(domain, codomain, mapping, derivative=derivative)

    def __matmul__(self, other: NonLinearOperator) -> NonLinearOperator:
        """Composes this operator with another: `(self @ other)(x) = self(other(x))`.

        The derivative of the composed operator is computed using the chain rule:
        `(F o G)'(x) = F'(G(x)) @ G'(x)`.

        Args:
            other: The operator to apply before this one.

        Returns:
            A new `NonLinearOperator` representing the composition.
        """

        if not isinstance(other, NonLinearOperator):
            raise TypeError("Operand must be a NonLinearOperator")

        domain = other.domain
        codomain = self.codomain

        def mapping(x: Any) -> Any:
            return self(other(x))

        if self._derivative is not None and other._derivative is not None:

            def derivative(x: Vector) -> LinearOperator:
                return self.derivative(other(x)) @ other.derivative(x)

        else:
            derivative = None

        return NonLinearOperator(domain, codomain, mapping, derivative=derivative)
