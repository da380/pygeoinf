"""Shared fixtures and example spaces for the v2 test suite.

**Threads.** The suite runs with the BLAS and OpenMP thread pools capped,
by :func:`_cap_threads` below, at ``PYGEOINF2_TEST_THREADS`` (default two).
Sixteen BLAS threads on a laptop make a seconds-long test swamp the
machine for no gain, and a test that measures iteration counts or wall
time should not depend on how many cores the runner has. A test that needs
more threads, or exactly one, says so with ``threadpool_limits`` of its
own; a test of the worker pools (``n_jobs``) sets its own
``inner_max_num_threads``, which wins over this cap inside the workers.
The same rule applies to anything run by hand next to the suite: one heavy
job at a time, with ``OMP_NUM_THREADS`` set, so that two runs do not
contend for the same cores and inflate each other's timings.
"""

from __future__ import annotations

import os
import sys
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


@pytest.fixture(scope="session", autouse=True)
def _cap_threads():
    """Cap the BLAS and OpenMP pools for the whole session; see the module
    docstring. ``PYGEOINF2_TEST_THREADS=0`` lifts the cap."""
    limit = int(os.environ.get("PYGEOINF2_TEST_THREADS", "2"))
    if limit <= 0:
        yield
        return
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=limit):
        yield


@pytest.fixture(autouse=True)
def _close_figures():
    """Every test ends with no open figures, so none leaks into the next.

    The examples and tutorials are run whole and leave their figures open,
    and the sibling package ``pygeoinf`` draws on ``plt.gca()``: a figure
    left over from here fails its tests, which run after these. pyplot is
    touched only if something has already imported it.
    """
    yield
    plt = sys.modules.get("matplotlib.pyplot")
    if plt is not None:
        plt.close("all")


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
        from scipy.linalg import cho_factor

        self._gram = np.asarray(gram, dtype=float)
        self._chol = np.linalg.cholesky(self._gram)
        # Factorized once: with a fresh ``np.linalg.solve`` per call,
        # ``solve_gram_to_columns`` at dimension 2000 would be 2000 full
        # solves, hours, and the fixture would be useless for measuring
        # anything the metric enters.
        self._factor = cho_factor(self._gram)

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
        from scipy.linalg import cho_solve

        return cho_solve(self._factor, c)

    def apply_gram_to_columns(self, columns: np.ndarray, /) -> np.ndarray:
        return self._gram @ columns

    def solve_gram_to_columns(self, columns: np.ndarray, /) -> np.ndarray:
        from scipy.linalg import cho_solve

        return cho_solve(self._factor, columns)

    def gram_matrix(self) -> np.ndarray:
        return self._gram.copy()

    def gram_diagonal(self) -> np.ndarray:
        return np.diagonal(self._gram).copy()


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

    The off-diagonal is scaled by ``1 / sqrt(dim)``. Without that the root's
    strict lower triangle grows like ``sqrt(dim)`` against a unit diagonal and
    the Gram stops being positive definite in floating point: measured
    condition numbers were 6e5 at dim 100 and 6e13 at dim 300, and
    ``cholesky`` failed at 500 -- so every dense-metric test above a few
    hundred dimensions was testing roundoff. With the scaling the condition
    number is about 3 at every size up to 3000.
    """
    if dim == 3:
        # The original, kept exactly so tests written against it do not move.
        root = np.array(
            [[1.0, 0.0, 0.0], [0.4, 1.2, 0.0], [-0.3, 0.5, 0.9]],
        )
        return DenseMetricSpace(root @ root.T)
    generator = np.random.default_rng(20260828)
    root = np.eye(dim) + (0.3 / np.sqrt(dim)) * np.tril(
        generator.standard_normal((dim, dim)), -1
    )
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


def coastlines_are_cached() -> bool:
    """True when cartopy already has the Natural Earth land shapefile on disk.

    A test that needs the coastlines skips when it is not, rather than
    downloading on a fresh runner: a network fetch inside the suite is slow,
    may fail, and is not what the test is checking.
    """
    try:
        from cartopy.io import shapereader
    except ImportError:
        return False
    import unittest.mock

    with unittest.mock.patch("cartopy.io.Downloader.acquire_resource") as blocked:
        blocked.side_effect = AssertionError("would download")
        try:
            shapereader.natural_earth(
                resolution="110m", category="physical", name="land"
            )
        except Exception:
            return False
    return True
