"""
Base classes for function providers.

This module contains the abstract base classes that define the provider
interface for generating functions from various families.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from pygeoinf.interval.functions import Function


class FunctionProvider(ABC):
    """
    Abstract base class for function providers.

    Function providers create Function objects from various families
    with a composable, lazy approach. All providers require explicit
    space specification from the user.
    """

    def __init__(self, space):
        """
        Initialize provider.

        Args:
            space: Lebesgue instance (contains domain information)
        """
        if space is None:
            raise ValueError(
                f"Space must be provided to {self.__class__.__name__}. "
                "Space cannot be None."
            )
        self.space = space

    @property
    def domain(self):
        """Get the domain from the space."""
        return self.space.function_domain


class IndexedFunctionProvider(FunctionProvider):
    """
    Provider for functions that can be accessed by index.

    Useful for basis functions, orthogonal families, etc.
    """

    @abstractmethod
    def get_function_by_index(self, index: int, **kwargs) -> 'Function':
        """Get function by index."""
        pass

    def get_functions(self, indices: List[int], **kwargs) -> List['Function']:
        """Get multiple functions by indices."""
        return [self.get_function_by_index(i, **kwargs) for i in indices]

    def restrict(self, restricted_space):
        """
        Create a restricted version of this provider.

        Returns a new provider that generates functions on the restricted
        space by taking the restriction of functions from the original
        provider.

        Args:
            restricted_space: The target space to restrict to.

        Returns:
            RestrictedFunctionProvider: A provider that returns restricted
                versions of the original provider's functions.

        Example:
            >>> # Provider on [0, 1]
            >>> provider = NormalModesProvider(space_full, ...)
            >>> # Restrict to [0, 0.5]
            >>> provider_restricted = provider.restrict(space_lower)
            >>> # Gets restricted kernels
            >>> kernel = provider_restricted.get_function_by_index(0)
        """
        return RestrictedFunctionProvider(self, restricted_space)


class RestrictedFunctionProvider(IndexedFunctionProvider):
    """
    A provider that returns restricted versions of another provider's
    functions.

    This wraps an existing IndexedFunctionProvider and restricts all
    generated functions to a smaller domain.
    """

    def __init__(self, original_provider: IndexedFunctionProvider,
                 restricted_space):
        """
        Initialize restricted provider.

        Args:
            original_provider: The provider to restrict
            restricted_space: The target space for restrictions
        """
        super().__init__(restricted_space)
        self.original_provider = original_provider

    def get_function_by_index(self, index: int, **kwargs) -> 'Function':
        """
        Get restricted version of function at given index.

        Args:
            index: Function index
            **kwargs: Additional arguments passed to original provider

        Returns:
            Function: Restricted function on the restricted space
        """
        # Get function from original provider
        original_func = self.original_provider.get_function_by_index(
            index, **kwargs
        )
        # Restrict it to our space
        return original_func.restrict(self.space)


class ParametricFunctionProvider(FunctionProvider):
    """
    Provider for parametric function families.

    Functions are defined by a parameter dictionary.
    """

    @abstractmethod
    def get_function_by_parameters(self, parameters: Dict[str, Any],
                                   **kwargs) -> 'Function':
        """Get a function with specific parameters."""
        pass

    def get_function(self, parameters: Optional[Dict[str, Any]] = None,
                     **kwargs) -> 'Function':
        """Get function with given or default parameters."""
        if parameters is None:
            parameters = self.get_default_parameters()
        return self.get_function_by_parameters(parameters, **kwargs)

    @abstractmethod
    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for this family."""
        pass


class RandomFunctionProvider(FunctionProvider):
    """
    Provider for randomly generated functions from a family.
    """

    def __init__(self, space, random_state: Optional[int] = None):
        """Initialize with optional random seed."""
        super().__init__(space)
        self.rng = np.random.RandomState(random_state)

    @abstractmethod
    def sample_function(self, **kwargs) -> 'Function':
        """Sample a random function from the family."""
        pass

    def get_function(self, **kwargs) -> 'Function':
        """Default implementation delegates to sample_function."""
        return self.sample_function(**kwargs)


class NullFunctionProvider(IndexedFunctionProvider):
    """
    A trivial provider that returns the zero function for all indices.

    Useful as a placeholder or for testing.
    """

    def __init__(self, space):
        """Initialize with space."""
        super().__init__(space)

    def get_function_by_index(self, index: int, **kwargs) -> 'Function':
        """
        Return the zero function.

        Args:
            index: Ignored
            **kwargs: Ignored

        Returns:
            Function: The zero function on this space
        """
        return self.space.zero
