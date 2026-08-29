"""
What an operator knows about itself at a point.

``Operator.at(x)`` returns a :class:`Linearisation`: the value and the
derivative together, computed in one call where the operator can manage it.
That is the whole point of separating ``at`` from ``__call__`` — a PDE solve
usually yields both, while a line search wants only the value and must not be
charged for a Jacobian it will discard.

Scalar-valued operators return the second-order analogue,
:class:`QuadraticModel`, which is what a Newton or trust-region step consumes.

See DESIGN.md section 5.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .operators import AffineOperator, LinearFunctional, LinearOperator

__all__ = ["Linearisation", "QuadraticModel"]


# ``eq=False``, so both of these keep object identity for ``==`` and ``hash``.
# The generated field-wise versions cannot work here: the fields hold vectors,
# which on every array-backed space are NumPy arrays, so ``==`` returned an
# array and ``bool()`` of it raised, and the ``__hash__`` that ``frozen=True``
# generates alongside ``eq=True`` raised ``TypeError: unhashable type:
# 'numpy.ndarray'``. A linearisation is a record of one evaluation at one
# point; identity is the only equality it can honestly offer.
@dataclass(frozen=True, eq=False)
class Linearisation[X, Y]:
    """An operator's value and derivative at a point."""

    point: X
    value: Y
    derivative: "LinearOperator[X, Y]"

    def as_affine(self) -> "AffineOperator[X, Y]":
        """The affine operator ``x -> value + derivative(x - point)``."""
        from .operators import AffineOperator

        codomain = self.derivative.codomain
        translation = codomain.subtract(self.value, self.derivative(self.point))
        return AffineOperator(self.derivative, translation)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(derivative={self.derivative!r})"


@dataclass(frozen=True, eq=False)
class QuadraticModel[X](Linearisation[X, float]):
    """A scalar-valued operator's local quadratic model.

    The **derivative is the stored primitive and the gradient is derived**,
    because that is the direction the information actually flows: a numerical
    adjoint method produces the derivative, and recovering the gradient from it
    costs an application of the inverse metric. Storing the gradient instead
    would invite callers to supply a derivative array in its place, which is
    the classic error described in DESIGN.md section 5.6.

    The gradient is computed once, on first access, and cached.
    """

    derivative: "LinearFunctional[X]"
    hessian: "LinearOperator[X, X] | None" = None

    @property
    def gradient(self) -> X:
        """The Riesz representer of the derivative — a vector in the domain.

        Equal to ``derivative.adjoint(1.0)``. The adjoint is where the metric
        enters, and it is the only place it enters.
        """
        cached = self.__dict__.get("_gradient", _MISSING)
        if cached is _MISSING:
            cached = self.derivative.adjoint(1.0)
            object.__setattr__(self, "_gradient", cached)
        return cached

    @property
    def has_hessian(self) -> bool:
        """True when a Hessian was supplied."""
        return self.hessian is not None

    def __repr__(self) -> str:
        return (
            f"QuadraticModel(value={self.value!r}, "
            f"hessian={'set' if self.has_hessian else 'None'})"
        )


_MISSING: Any = object()
