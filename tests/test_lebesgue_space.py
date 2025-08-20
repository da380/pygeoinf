"""
Tests for the LebesgueSpace implementation.

This file demonstrates how to test a HilbertSpace that wraps an abstract
vector type, rather than using NumPy arrays directly.
"""

import pytest
import numpy as np
from typing import Callable
from pygeoinf.hilbert_space import LebesgueSpace, T_vec
from .checks.hilbert_space import HilbertSpaceChecks


# 1. Correct the custom vector class methods to operate on `self`.
class CustomVector:
    """A simple wrapper around a NumPy array to act as an abstract vector."""

    def __init__(self, data: np.ndarray):
        self.data = data

    def __add__(self, other: "CustomVector") -> "CustomVector":
        return CustomVector(self.data + other.data)

    def __sub__(self, other: "CustomVector") -> "CustomVector":
        return CustomVector(self.data - other.data)

    def __mul__(self, scalar: float) -> "CustomVector":
        return CustomVector(self.data * scalar)

    def __rmul__(self, scalar: float) -> "CustomVector":
        return self * scalar

    def copy(self) -> "CustomVector":
        """Returns a deep copy of this vector."""
        return CustomVector(self.data.copy())

    def ax(self, alpha: float) -> None:
        """In-place scaling: self := alpha * self."""
        self.data *= alpha

    def axpy(self, alpha: float, x: "CustomVector") -> None:
        """In-place scale and add: self := alpha * x + self."""
        self.data += alpha * x.data


def to_components(vec: CustomVector) -> np.ndarray:
    return vec.data


def from_components(arr: np.ndarray) -> CustomVector:
    return CustomVector(arr)


def custom_copy(vec: CustomVector) -> CustomVector:
    """Standalone copy function."""
    return vec.copy()


def custom_ax(alpha: float, x: CustomVector) -> None:
    """Standalone ax function that calls the method on the vector."""
    x.ax(alpha)


def custom_axpy(alpha: float, x: CustomVector, y: CustomVector) -> None:
    """Standalone axpy function. This operation modifies y."""
    y.axpy(alpha, x)


@pytest.fixture
def space() -> LebesgueSpace[CustomVector]:
    """Provides a 5-dimensional LebesgueSpace instance for the tests."""
    return LebesgueSpace(
        5,
        to_components,
        from_components,
        ax=custom_ax,
        axpy=custom_axpy,
    )


class TestLebesgueSpace(HilbertSpaceChecks):
    """
    Runs the standard suite of Hilbert space checks on the LebesgueSpace class
    wrapping a custom vector type.
    """

    def test_inner_product_is_dot_product_of_components(
        self, space: LebesgueSpace, x: CustomVector, y: CustomVector
    ):
        """
        Confirms that <x, y> is calculated as dot(to_components(x), to_components(y)).
        """
        inner_product_val = space.inner_product(x, y)
        x_comp = space.to_components(x)
        y_comp = space.to_components(y)
        expected_val = np.dot(x_comp, y_comp)
        assert np.isclose(inner_product_val, expected_val)
