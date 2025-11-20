"""Base classes for operators on interval domains."""

from abc import ABC, abstractmethod

from pygeoinf.linear_operators import LinearOperator

from ..functions import Function


class SpectralOperator(LinearOperator, ABC):
    """
    Abstract base class for spectral operators on interval domains.

    Provides common functionality for eigenvalue/eigenfunction access
    and error handling.
    """

    def __init__(
        self,
        domain,
        codomain,
        mapping
    ):
        super().__init__(domain, codomain, mapping)

    @abstractmethod
    def get_eigenvalue(self, index: int) -> float:
        """Get the eigenvalue at a specific index."""
        pass

    @abstractmethod
    def get_eigenfunction(self, index: int) -> Function:
        """Get the eigenfunction at a specific index."""
        pass

    @abstractmethod
    def _apply(self, f: Function) -> Function:
        """Apply the operator to a function."""
        pass
