"""
Module for abstract helper class for function spaces defined on symmetric spaces.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Any, List, Optional
import numpy as np

from pygeoinf.hilbert_space import HilbertSpace, EuclideanSpace
from pygeoinf.operators import LinearOperator
from pygeoinf.linear_forms import LinearForm
from pygeoinf.gaussian_measure import GaussianMeasure


class SymmetricSpaceHelper:
    """
    An abstract base class for

    This class can be inherited by a Hilbert space to provide common functionality for
    functions on symmetric spaces.
    """

    @abstractmethod
    def random_point(self) -> Any:
        """Returns a single random point from the underlying symmetric space."""

    def random_points(self, n: int) -> List[Any]:
        """
        Returns a list of `n` random points.

        Args:
            n: The number of random points to generate.
        """
        return [self.random_point() for _ in range(n)]

    @abstractmethod
    def laplacian(self) -> LinearOperator:
        """
        Returns the Laplacian as an automorphism on the space. Strictly, this
        operator is well-defined only on a dense subset of the true function space
        but this is ignored numerically.
        """
        pass
