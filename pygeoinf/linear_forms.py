"""
Provides the `LinearForm` class to represent linear functionals.

A linear form is a linear mapping from a vector in a Hilbert space to a
scalar (a real number). This class provides a concrete representation for
elements of the dual space of a `HilbertSpace`.

A `LinearForm` can be thought of as a dual vector and is a fundamental component
for defining inner products and adjoint operators within the library.
"""

from __future__ import annotations
from typing import Callable, Optional, Any, TYPE_CHECKING

import numpy as np

# This block only runs for type checkers, not at runtime
if TYPE_CHECKING:
    from .hilbert_space import HilbertSpace, EuclideanSpace
    from .operators import LinearOperator


class LinearForm:
    """
    Represents a linear form, a functional that maps vectors to scalars.

    A `LinearForm` is an element of a dual `HilbertSpace`. It is defined by its
    action on vectors from its `domain` space. Internally, this action is
    represented by a component vector, which when dotted with the component
    vector of a primal space element, produces the scalar result.
    """

    def __init__(
        self,
        domain: HilbertSpace,
        /,
        *,
        mapping: Optional[Callable[[Any], float]] = None,
        components: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initializes the LinearForm.

        A form can be defined either by its functional mapping or directly
        by its component vector. If a mapping is provided without components,
        the components will be computed by evaluating the mapping on the
        basis vectors of the domain.

        Args:
            domain: The Hilbert space on which the form is defined.
            mapping: A function `f(x)` defining the action of the form.
            components: The component representation of the form.
        """

        self._domain: HilbertSpace = domain

        if components is None:
            if mapping is None:
                raise AssertionError("Neither mapping nor components specified.")
            self._compute_components(mapping)
        else:
            self._components: np.ndarray = components

    @staticmethod
    def from_linear_operator(operator: "LinearOperator") -> LinearForm:
        """
        Creates a LinearForm from an operator that maps to a 1D Euclidean space.
        """
        from .hilbert_space import EuclideanSpace

        assert operator.codomain == EuclideanSpace(1)
        return LinearForm(operator.domain, mapping=lambda x: operator(x)[0])

    @property
    def domain(self) -> HilbertSpace:
        """The Hilbert space on which the form is defined."""
        return self._domain

    @property
    def components(self) -> np.ndarray:
        """
        The component vector of the form.
        """
        return self._components

    @property
    def as_linear_operator(self) -> "LinearOperator":
        """
        Represents the linear form as a `LinearOperator`.

        The resulting operator maps from the form's original domain to a
        1-dimensional `EuclideanSpace`, where the single component of the output
        is the scalar result of the form's action.
        """
        from .hilbert_space import EuclideanSpace
        from .operators import LinearOperator

        return LinearOperator(
            self.domain,
            EuclideanSpace(1),
            lambda x: np.array([self(x)]),
            dual_mapping=lambda y: y * self,
        )

    def copy(self) -> LinearForm:
        """
        Creates a deep copy of the linear form.
        """
        return LinearForm(self.domain, components=self.components.copy())

    def __call__(self, x: Any) -> float:
        """Applies the linear form to a vector."""
        return np.dot(self._components, self.domain.to_components(x))

    def __neg__(self) -> LinearForm:
        """Returns the additive inverse of the form."""
        return LinearForm(self.domain, components=-self._components)

    def __mul__(self, a: float) -> LinearForm:
        """Returns the product of the form and a scalar."""
        return LinearForm(self.domain, components=a * self._components)

    def __rmul__(self, a: float) -> LinearForm:
        """Returns the product of the form and a scalar."""
        return self * a

    def __truediv__(self, a: float) -> LinearForm:
        """Returns the division of the form by a scalar."""
        return self * (1.0 / a)

    def __add__(self, other: LinearForm) -> LinearForm:
        """Returns the sum of this form and another."""
        return LinearForm(self.domain, components=self.components + other.components)

    def __sub__(self, other: LinearForm) -> LinearForm:
        """Returns the difference between this form and another."""
        return LinearForm(self.domain, components=self.components - other.components)

    def __imul__(self, a: float) -> "LinearForm":
        """
        Performs in-place scalar multiplication: self *= a.
        """
        self._components *= a
        return self

    def __iadd__(self, other: "LinearForm") -> "LinearForm":
        """
        Performs in-place addition with another form: self += other.
        """
        if self.domain != other.domain:
            raise ValueError("Linear forms must share the same domain for addition.")
        self._components += other.components
        return self

    def __str__(self) -> str:
        """Returns the string representation of the form's components."""
        return self.components.__str__()

    def _compute_components(self, mapping: Callable[[Any], float]):
        """
        Computes the component vector of the form.
        """
        self._components = np.zeros(self.domain.dim)
        cx = np.zeros(self.domain.dim)
        for i in range(self.domain.dim):
            cx[i] = 1
            x = self.domain.from_components(cx)
            self._components[i] = mapping(x)
            cx[i] = 0
