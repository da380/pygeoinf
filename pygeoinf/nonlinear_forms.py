"""
Provides the `NonLinearForm` base class to represent non-linear functionals.

A non-linear form, or functional, is a mapping from a vector in a Hilbert
space to a scalar. This class provides a foundational structure for these
functionals, equipping them with algebraic operations and an interface for
derivatives like gradients and Hessians.
"""

from __future__ import annotations
from typing import Callable, Optional, Any, TYPE_CHECKING


# This block only runs for type checkers, not at runtime
if TYPE_CHECKING:
    from .hilbert_space import HilbertSpace, Vector
    from .linear_forms import LinearForm
    from .linear_operators import LinearOperator


class NonLinearForm:
    """
    Represents a general non-linear functional that maps vectors to scalars.

    This class serves as the foundation for all forms. It defines the basic
    callable interface `form(x)` and overloads arithmetic operators (`+`, `-`, `*`)
    to create new forms. It also provides an optional framework for specifying
    a form's gradient and Hessian.
    """

    def __init__(
        self,
        domain: HilbertSpace,
        mapping: Callable[[Vector], float],
        /,
        *,
        gradient: Optional[Callable[[Vector], Vector]] = None,
        hessian: Optional[Callable[[Vector], LinearOperator]] = None,
    ) -> None:
        """
        Initializes the NonLinearForm.

        Args:
            domain: The Hilbert space on which the form is defined.
            mapping: The function `f(x)` that defines the action of the form.
            gradient: An optional function that computes the gradient of the form.
            hessian: An optional function that computes the Hessian of the form.
        """

        self._domain: HilbertSpace = domain
        self._mapping = mapping
        self._gradient = gradient
        self._hessian = hessian

    @property
    def domain(self) -> HilbertSpace:
        """The Hilbert space on which the form is defined."""
        return self._domain

    @property
    def has_gradient(self) -> bool:
        """True if the form has a gradient."""
        return self._gradient is not None

    @property
    def has_hessian(self) -> bool:
        """True if the form has a Hessian."""
        return self._hessian is not None

    def __call__(self, x: Any) -> float:
        """Applies the linear form to a vector."""
        return self._mapping(x)

    def gradient(self, x: Any) -> Vector:
        """
        Computes the gradient of the form at a given point.

        Args:
            x: The vector at which to evaluate the gradient.

        Returns:
            The gradient of the form as a vector in the domain space.

        Raises:
            NotImplementedError: If a gradient function was not provided
                during initialization.
        """
        if self._gradient is None:
            raise NotImplementedError("Gradient not implemented for this form.")
        return self._gradient(x)

    def derivative(self, x: Vector) -> LinearForm:
        """
        Computes the derivative of the form at a given point.

        Args:
            x: The vector at which to evaluate the derivative.

        Returns:
            The derivative of the form as a `LinearForm`.

        Raises:
            NotImplementedError: If a gradient function was not provided
                during initialization.
        """
        return self.domain.to_dual(self.gradient(x))

    def hessian(self, x: Any) -> LinearOperator:
        """
        Computes the Hessian of the form at a given point.

        Args:
            x: The vector at which to evaluate the Hessian.

        Returns:
            The Hessian of the form as a LinearOperator mapping the domain to itself.

        Raises:
            NotImplementedError: If a Hessian function was not provided
                during initialization.
        """
        if self._hessian is None:
            raise NotImplementedError("Hessian not implemented for this form.")
        return self._hessian(x)

    def __neg__(self) -> NonLinearForm:
        """Returns the additive inverse of the form."""

        if self._gradient is None:
            gradient = None
        else:

            def gradient(x: Vector) -> Vector:
                return self.domain.negative(self.gradient(x))

        if self._hessian is None:
            hessian = None
        else:

            def hessian(x: Vector) -> LinearOperator:
                return -self.hessian(x)

        return NonLinearForm(
            self.domain, lambda x: -self(x), gradient=gradient, hessian=hessian
        )

    def __mul__(self, a: float) -> NonLinearForm:
        """Returns the product of the form and a scalar."""

        if self._gradient is None:
            gradient = None
        else:

            def gradient(x: Vector) -> Vector:
                return self.domain.multiply(a, self.gradient(x))

        if self._hessian is None:
            hessian = None
        else:

            def hessian(x: Vector) -> LinearOperator:
                return a * self.hessian(x)

        return NonLinearForm(
            self.domain,
            lambda x: a * self(x),
            gradient=gradient,
            hessian=hessian,
        )

    def __rmul__(self, a: float) -> NonLinearForm:
        """Returns the product of the form and a scalar."""
        return self * a

    def __truediv__(self, a: float) -> NonLinearForm:
        """Returns the division of the form by a scalar."""
        return self * (1.0 / a)

    def __add__(self, other: NonLinearForm) -> NonLinearForm:
        """Returns the sum of this form and another."""

        if self._gradient is None or other._gradient is None:
            gradient = None
        else:

            def gradient(x: Vector) -> Vector:
                return self.domain.add(self.gradient(x), other.gradient(x))

        if self._hessian is None or other._hessian is None:
            hessian = None
        else:

            def hessian(x: Vector) -> LinearOperator:
                return self.hessian(x) + other.hessian(x)

        return NonLinearForm(
            self.domain,
            lambda x: self(x) + other(x),
            gradient=gradient,
            hessian=hessian,
        )

    def __sub__(self, other: NonLinearForm) -> NonLinearForm:
        """Returns the difference between this form and another."""

        if self._gradient is None or other._gradient is None:
            gradient = None
        else:

            def gradient(x: Vector) -> Vector:
                return self.domain.subtract(self.gradient(x), other.gradient(x))

        if self._hessian is None or other._hessian is None:
            hessian = None
        else:

            def hessian(x: Vector) -> LinearOperator:
                return self.hessian(x) - other.hessian(x)

        return NonLinearForm(
            self.domain,
            lambda x: self(x) - other(x),
            gradient=gradient,
            hessian=hessian,
        )
