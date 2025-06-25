"""
Module containing the base class for Sobolev spaces defined on homogeneous spaces. 
"""

from abc import ABC, abstractmethod


class homogeneous_space(ABC):

    def __init__(self, exponent, scale):
        """
        Args:
            exponent (float): The Sobolev exponent.
            scale (float): The Sobolev scale.

        Notes:

        The scale is defined relative to a characteristic
        length for the domain. It must be positive.
        """

        if scale <= 0:
            raise ValueError("Scale must be positive")

        self._exponent = exponent
        self._scale = scale

    @property
    def exponent(self):
        return self._exponent
