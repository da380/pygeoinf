"""
Provides the `AffineOperator` class for affine mappings between Hilbert spaces.
"""

from __future__ import annotations
from typing import Any

from .checks.affine_operators import AffineOperatorAxiomChecks
from .linear_operators import LinearOperator
from .nonlinear_operators import NonLinearOperator


class AffineOperator(AffineOperatorAxiomChecks, NonLinearOperator):
    """
    Represents an affine transformation between two Hilbert spaces.

    An affine operator is a mapping F(x) = A(x) + b, where 'A' is a bounded
    linear operator and 'b' is a fixed translation vector in the codomain.
    """

    def __init__(
        self,
        linear_part: LinearOperator,
        translation: Any,
    ) -> None:
        """
        Initializes the AffineOperator.

        Args:
            linear_part: The underlying linear operator 'A'.
            translation: The translation vector 'b' in the operator's codomain.
                If the translation is the zero vector, this operator is
                equivalent to the linear part.

        Raises:
            TypeError: If the translation vector is not an element of the
                linear operator's codomain.
        """
        if not linear_part.codomain.is_element(translation):
            raise TypeError(
                "The translation vector must be an element of the linear "
                "operator's codomain."
            )

        self._linear_part = linear_part
        self._translation = translation

        domain = linear_part.domain
        codomain = linear_part.codomain

        def mapping(x: Any) -> Any:
            # F(x) = A(x) + b
            return codomain.add(self._linear_part(x), self._translation)

        def derivative(x: Any) -> LinearOperator:
            # The Fréchet derivative of A(x) + b is just A
            return self._linear_part

        super().__init__(domain, codomain, mapping, derivative=derivative)

    @property
    def linear_part(self) -> LinearOperator:
        """The underlying linear mapping 'A'."""
        return self._linear_part

    @property
    def translation_part(self) -> Any:
        """The translation vector 'b'."""
        return self._translation

    # ------------------------------------------------------------------- #
    #                      Algebraic Overloads                            #
    # ------------------------------------------------------------------- #

    def __add__(self, other: Any) -> Any:
        if isinstance(other, AffineOperator):
            if self.domain != other.domain or self.codomain != other.codomain:
                raise ValueError("Domains and codomains must match for addition.")
            new_linear = self.linear_part + other.linear_part
            new_translation = self.codomain.add(
                self.translation_part, other.translation_part
            )
            return AffineOperator(new_linear, new_translation)

        elif isinstance(other, LinearOperator):
            if self.domain != other.domain or self.codomain != other.codomain:
                raise ValueError("Domains and codomains must match for addition.")
            # F(x) + L(x) = (A + L)x + a
            new_linear = self.linear_part + other
            return AffineOperator(new_linear, self.translation_part)

        return super().__add__(other)

    def __radd__(self, other: Any) -> Any:
        if isinstance(other, LinearOperator):
            # Addition is commutative: L(x) + F(x) = F(x) + L(x)
            return self.__add__(other)
        return super().__radd__(other)

    def __sub__(self, other: Any) -> Any:
        if isinstance(other, AffineOperator):
            if self.domain != other.domain or self.codomain != other.codomain:
                raise ValueError("Domains and codomains must match for subtraction.")
            new_linear = self.linear_part - other.linear_part
            new_translation = self.codomain.subtract(
                self.translation_part, other.translation_part
            )
            return AffineOperator(new_linear, new_translation)

        elif isinstance(other, LinearOperator):
            if self.domain != other.domain or self.codomain != other.codomain:
                raise ValueError("Domains and codomains must match for subtraction.")
            # F(x) - L(x) = (A - L)x + a
            new_linear = self.linear_part - other
            return AffineOperator(new_linear, self.translation_part)

        return super().__sub__(other)

    def __rsub__(self, other: Any) -> Any:
        if isinstance(other, LinearOperator):
            if self.domain != other.domain or self.codomain != other.codomain:
                raise ValueError("Domains and codomains must match for subtraction.")
            # L(x) - F(x) = (L - A)x - a
            new_linear = other - self.linear_part
            new_translation = self.codomain.negative(self.translation_part)
            return AffineOperator(new_linear, new_translation)
        return super().__rsub__(other)

    def __mul__(self, alpha: float) -> "AffineOperator":
        if not isinstance(alpha, (int, float)):
            return NotImplemented
        new_linear = self.linear_part * alpha
        new_translation = self.codomain.multiply(alpha, self.translation_part)
        return AffineOperator(new_linear, new_translation)

    def __rmul__(self, alpha: float) -> "AffineOperator":
        return self.__mul__(alpha)

    def __matmul__(self, other: Any) -> Any:
        if isinstance(other, AffineOperator):
            if self.domain != other.codomain:
                raise ValueError(
                    "Codomain of right operator must match domain of left operator."
                )
            # F(G(x)) = (A @ B)x + (A(b) + a)
            new_linear = self.linear_part @ other.linear_part
            A_b = self.linear_part(other.translation_part)
            new_translation = self.codomain.add(A_b, self.translation_part)
            return AffineOperator(new_linear, new_translation)

        elif isinstance(other, LinearOperator):
            if self.domain != other.codomain:
                raise ValueError(
                    "Codomain of right operator must match domain of left operator."
                )
            # F(L(x)) = (A @ L)x + a
            new_linear = self.linear_part @ other
            return AffineOperator(new_linear, self.translation_part)

        return super().__matmul__(other)

    def __rmatmul__(self, other: Any) -> Any:
        if isinstance(other, LinearOperator):
            if other.domain != self.codomain:
                raise ValueError(
                    "Codomain of right operator must match domain of left operator."
                )
            # L(F(x)) = (L @ A)x + L(a)
            new_linear = other @ self.linear_part
            new_translation = other(self.translation_part)
            return AffineOperator(new_linear, new_translation)

        # We don't override for non-linear operators, let the base class handle it
        return super().__rmatmul__(other)
