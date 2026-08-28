"""Shared fixtures and example spaces for the v2 test suite."""

from __future__ import annotations

from typing import Hashable

import numpy as np
import pytest
from numpy.random import default_rng

from pygeoinf2.algebra.spaces import (
    ArrayVectorMixin,
    CoordinateSpace,
    DiagonalMetricSpace,
)

SEED = 20260826


@pytest.fixture
def rng():
    return default_rng(SEED)


class WeightedSpace(ArrayVectorMixin, DiagonalMetricSpace[np.ndarray]):
    """A non-orthonormal coordinate space, for exercising the Gram matrix.

    Vectors are their own component arrays, so nothing but the metric
    distinguishes this from a Euclidean space of the same dimension. That is
    what makes it a useful test: on an orthonormal basis every derivative/
    gradient confusion is invisible.
    """

    def __init__(self, metric_values: np.ndarray) -> None:
        super().__init__(metric_values)

    def _key(self) -> Hashable:
        return tuple(self.metric_values.tolist())

    def to_components(self, x: np.ndarray) -> np.ndarray:
        return x

    def from_components(self, c: np.ndarray) -> np.ndarray:
        return c


class DenseMetricSpace(ArrayVectorMixin, CoordinateSpace[np.ndarray]):
    """A coordinate space with a full, non-diagonal Gram matrix.

    Exercises the generic ``CoordinateSpace`` paths — the dense Cholesky in
    ``white_noise_components`` in particular — which the diagonal and
    orthonormal subclasses bypass.
    """

    def __init__(self, gram: np.ndarray) -> None:
        self._gram = np.asarray(gram, dtype=float)
        self._chol = np.linalg.cholesky(self._gram)

    @property
    def dim(self) -> int:
        return self._gram.shape[0]

    def _key(self) -> Hashable:
        return tuple(self._gram.ravel().tolist())

    def to_components(self, x: np.ndarray) -> np.ndarray:
        return x

    def from_components(self, c: np.ndarray) -> np.ndarray:
        return c

    def apply_gram(self, c: np.ndarray) -> np.ndarray:
        return self._gram @ c

    def solve_gram(self, c: np.ndarray) -> np.ndarray:
        return np.linalg.solve(self._gram, c)


def make_weighted_space() -> WeightedSpace:
    return WeightedSpace(np.array([1.0, 4.0, 9.0, 0.25]))


def make_dense_metric_space(dim: int = 3) -> DenseMetricSpace:
    """A space whose Gram matrix is dense, at any dimension.

    The size is a parameter so this can stand in for
    :func:`make_weighted_space` wherever a test fixes a dimension -- which is
    what the metric rule needs, since only a non-diagonal Gram distinguishes
    metric-correct code from code that merely agrees with the components.

    Built from a lower-triangular root with a strong diagonal, so the Gram is
    positive definite and well conditioned: the point of the fixture is that
    the metric is not diagonal, not that floating point is hard.
    """
    if dim == 3:
        # The original, kept exactly so tests written against it do not move.
        root = np.array(
            [[1.0, 0.0, 0.0], [0.4, 1.2, 0.0], [-0.3, 0.5, 0.9]],
        )
        return DenseMetricSpace(root @ root.T)
    generator = np.random.default_rng(20260828)
    root = np.eye(dim) + 0.3 * np.tril(generator.standard_normal((dim, dim)), -1)
    return DenseMetricSpace(root @ root.T)


def values(space, *fields):
    """Grid values of one or more fields, for comparing with plain numpy.

    A sphere's vectors are ``SHGrid`` objects, so ``np.allclose(a, b)`` on two
    of them raises rather than comparing. Where a test is checking numbers
    against an independently computed array this is the honest unwrapping;
    where it is checking two vectors of the same space against each other,
    ``space.norm(space.subtract(a, b))`` says it better.
    """
    unwrapped = [space.grid_values(field) for field in fields]
    return unwrapped[0] if len(unwrapped) == 1 else unwrapped
