"""
Module defining the LinearForm class.
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
    Represents a linear form, which is a linear functional that maps
    vectors from a Hilbert space to a scalar value (a real number).
    """

    def __init__(
        self,
        domain: "HilbertSpace",
        /,
        *,
        mapping: Optional[Callable[[Any], float]] = None,
        components: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initializes the LinearForm.

        A form can be defined either by its mapping or its component vector.

        Args:
            domain (HilbertSpace): The Hilbert space on which the form is defined.
            mapping (callable, optional): A function defining the action of the form.
            components (np.ndarray, optional): The component representation of
                the form.
        """

        self._domain: "HilbertSpace" = domain
        self._components: Optional[np.ndarray] = components
        self._mapping: Callable[[Any], float]

        if components is None:
            if mapping is None:
                raise AssertionError("Neither mapping nor components specified.")
            else:
                self._mapping = mapping
        else:
            if mapping is None:
                self._mapping = self._mapping_from_components
            else:
                self._mapping = mapping

    @staticmethod
    def from_linear_operator(operator: "LinearOperator") -> "LinearForm":
        """
        Creates a LinearForm from an operator that maps to a 1D Euclidean space.
        """
        from .hilbert_space import EuclideanSpace

        assert operator.codomain == EuclideanSpace(1)
        return LinearForm(operator.domain, mapping=lambda x: operator(x)[0])

    @property
    def domain(self) -> "HilbertSpace":
        """The Hilbert space on which the form is defined."""
        return self._domain

    @property
    def components_stored(self) -> bool:
        """True if the form's component vector is cached."""
        return self._components is not None

    @property
    def components(self) -> np.ndarray:
        """
        The component vector of the form.

        The components are computed and cached on first access if not
        provided during initialization.
        """
        if self.components_stored:
            return self._components
        else:
            self.store_components()
            return self.components

    def store_components(self) -> None:
        """Computes and caches the component vector of the form."""
        if not self.components_stored:
            self._components = np.zeros(self.domain.dim)
            cx = np.zeros(self.domain.dim)
            for i in range(self.domain.dim):
                cx[i] = 1
                x = self.domain.from_components(cx)
                self._components[i] = self(x)
                cx[i] = 0

    @property
    def as_linear_operator(self) -> "LinearOperator":
        """
        Represents the linear form as a LinearOperator mapping to a
        1D Euclidean space.
        """
        from .hilbert_space import EuclideanSpace
        from .operators import LinearOperator

        return LinearOperator(
            self.domain,
            EuclideanSpace(1),
            lambda x: np.array([self(x)]),
            dual_mapping=lambda y: y * self,
        )

    def __call__(self, x: Any) -> float:
        """Applies the linear form to a vector."""
        return self._mapping(x)

    def __neg__(self) -> "LinearForm":
        """Returns the additive inverse of the form."""
        if self.components_stored:
            return LinearForm(self.domain, components=-self._components)
        else:
            return LinearForm(self.domain, mapping=lambda x: -self(x))

    def __mul__(self, a: float) -> "LinearForm":
        """Returns the product of the form and a scalar."""
        if self.components_stored:
            return LinearForm(self.domain, components=a * self._components)
        else:
            return LinearForm(self.domain, mapping=lambda x: a * self(x))

    def __rmul__(self, a: float) -> "LinearForm":
        """Returns the product of the form and a scalar."""
        return self * a

    def __truediv__(self, a: float) -> "LinearForm":
        """Returns the division of the form by a scalar."""
        return self * (1.0 / a)

    def __add__(self, other: "LinearForm") -> "LinearForm":
        """Returns the sum of this form and another."""
        if self.components_stored and other.components_stored:
            return LinearForm(
                self.domain, components=self.components + other.components
            )
        else:
            return LinearForm(self.domain, mapping=lambda x: self(x) + other(x))

    def __sub__(self, other: "LinearForm") -> "LinearForm":
        """Returns the difference between this form and another."""
        if self.components_stored and other.components_stored:
            return LinearForm(
                self.domain, components=self.components - other.components
            )
        else:
            return LinearForm(self.domain, mapping=lambda x: self(x) - other(x))

    def __str__(self) -> str:
        """Returns the string representation of the form's components."""
        return self.components.__str__()

    def _mapping_from_components(self, x: Any) -> float:
        """Implements the action of the form using its cached components."""
        return np.dot(self._components, self.domain.to_components(x))
