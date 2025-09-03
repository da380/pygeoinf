"""
Provides the `NonLinearForm` class to represent non-linear functionals.

A non-linear form is a  mapping from a  Hilbert space to a scalar (a real number). 
"""

from __future__ import annotations
from typing import Callable, Optional, Any, TYPE_CHECKING

import numpy as np

# This block only runs for type checkers, not at runtime
if TYPE_CHECKING:
    from .hilbert_space import HilbertSpace, EuclideanSpace, Vector
    from .linear_forms import LinearForm
    from .linear_operators import LinearOperator


class NonLinearForm:
    """
    Represents a non-linear form, a functional that maps vectors to scalars.
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
            mapping: A function defining the action of the form.
            gradient: A function defining the gradient of the form.
            hessian: A function defining the Hessian of the form.
        """

        self._domain: HilbertSpace = domain
        self._mapping = mapping
        self._gradient = gradient
        self._hessian = hessian

    @property
    def domain(self) -> HilbertSpace:
        """The Hilbert space on which the form is defined."""
        return self._domain

    def __call__(self, x: Any) -> float:
        """Applies the linear form to a vector."""
        return self._mapping(x)

    def gradient(self, x: Any) -> Vector:
        """Computes the gradient of the form at a point."""
        if self._gradient is None:
            raise NotImplementedError("Gradient not implemented for this form.")
        return self._gradient(x)

    def hessian(self, x: Any) -> LinearOperator:
        """Computes the Hessian of the form at a point."""
        if self._hessian is None:
            raise NotImplementedError("Hessian not implemented for this form.")
        return self._hessian(x)

    def __neg__(self) -> NonLinearForm:
        """Returns the additive inverse of the form."""
        return NonLinearForm(self.domain, mapping=lambda x: -self(x))

    def __mul__(self, a: float) -> LinearForm:
        """Returns the product of the form and a scalar."""
        return NonLinearForm(self.domain, mapping=lambda x: a * self(x))

    def __rmul__(self, a: float) -> LinearForm:
        """Returns the product of the form and a scalar."""
        return self * a

    def __truediv__(self, a: float) -> LinearForm:
        """Returns the division of the form by a scalar."""
        return self * (1.0 / a)

    def __add__(self, other: LinearForm) -> LinearForm:
        """Returns the sum of this form and another."""
        return NonLinearForm(self.domain, mapping=lambda x: self(x) + other(x))

    def __sub__(self, other: LinearForm) -> LinearForm:
        """Returns the difference between this form and another."""
        return NonLinearForm(self.domain, mapping=lambda x: self(x) - other(x))
