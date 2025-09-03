"""
Module containing a common base class for both linear and non-linear operators
between Hilbert spaces. 
"""

from __future__ import annotations
from typing import Callable, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .hilbert_space import HilbertSpace


class Operator:
    """
    A base class for operators between two Hilbert spaces.
    """

    def __init__(
        self,
        domain: HilbertSpace,
        codomain: HilbertSpace,
        mapping: Callable[[Any], Any],
    ) -> None:
        """
        Initializes the Operator.

        Args:
            domain (HilbertSpace): Domain of the operator.
            codomain (HilbertSpace): Codomain of the operator.
            mapping (callable): The function defining the mapping from the
                domain to the codomain.
        """
        self._domain: HilbertSpace = domain
        self._codomain: HilbertSpace = codomain
        self.__mapping: Callable[[Any], Any] = mapping

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

    def __call__(self, x: Any) -> Any:
        """Applies the operator's mapping to a vector."""
        return self.__mapping(x)
