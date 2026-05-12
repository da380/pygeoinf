"""Tests for Gaussian credible subsets."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import chi2

from pygeoinf.gaussian_measure import GaussianMeasure
from pygeoinf.hilbert_space import (
    EuclideanSpace,
    HilbertSpace,
    MassWeightedHilbertSpace,
)
from pygeoinf.linear_forms import LinearForm
from pygeoinf.subsets import Ball, Ellipsoid


class ZeroDimensionalSpace(HilbertSpace):
    """Minimal basis-free-style Hilbert space for rank validation tests."""

    @property
    def dim(self) -> int:
        return 0

    def to_dual(self, x: np.ndarray) -> LinearForm:
        return LinearForm(self, components=np.array([]))

    def from_dual(self, xp: LinearForm) -> np.ndarray:
        return np.array([])

    def to_components(self, x: np.ndarray) -> np.ndarray:
        return x

    def from_components(self, c: np.ndarray) -> np.ndarray:
        return c

    def is_element(self, x) -> bool:
        return isinstance(x, np.ndarray) and x.shape == (0,)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ZeroDimensionalSpace)


def test_credible_set_returns_covariance_ellipsoid():
    """A full-rank Gaussian returns the chi-square Mahalanobis ellipsoid."""
    space = EuclideanSpace(2)
    mean = np.array([1.0, -2.0])
    covariance = np.diag([4.0, 9.0])
    probability = 0.8

    measure = GaussianMeasure.from_covariance_matrix(
        space, covariance, expectation=mean
    )

    credible_set = measure.credible_set(probability)

    radius = np.sqrt(chi2.ppf(probability, df=space.dim))
    assert isinstance(credible_set, Ellipsoid)
    assert_allclose(credible_set.center, mean, rtol=0.0, atol=0.0)
    assert_allclose(credible_set.radius, radius, rtol=1e-12, atol=1e-12)

    boundary_point = mean + np.array([2.0 * radius, 0.0])
    outside_point = mean + np.array([2.0 * radius * 1.01, 0.0])

    assert credible_set.is_element(boundary_point, rtol=1e-10)
    assert not credible_set.is_element(outside_point, rtol=1e-10)


def test_credible_set_support_function_uses_covariance_shape():
    """The returned ellipsoid exposes the covariance-shaped support function."""
    space = EuclideanSpace(2)
    mean = np.array([1.0, -2.0])
    covariance = np.array([[4.0, 1.5], [1.5, 2.25]])
    probability = 0.5

    measure = GaussianMeasure.from_covariance_matrix(
        space, covariance, expectation=mean
    )
    credible_set = measure.credible_set(probability)

    radius = np.sqrt(chi2.ppf(probability, df=space.dim))
    q = np.array([0.3, -1.2])
    expected_support = q @ mean + radius * np.sqrt(q @ covariance @ q)

    assert_allclose(
        credible_set.support_function(q), expected_support, rtol=1e-12, atol=1e-12
    )


def test_credible_set_can_return_cameron_martin_ball():
    """The same confidence region can be returned as a ball in CM geometry."""
    space = EuclideanSpace(2)
    mean = np.array([1.0, -2.0])
    covariance = np.diag([4.0, 9.0])
    probability = 0.9

    measure = GaussianMeasure.from_covariance_matrix(
        space, covariance, expectation=mean
    )
    credible_ball = measure.credible_set(probability, geometry="cameron_martin")

    radius = np.sqrt(chi2.ppf(probability, df=space.dim))
    assert isinstance(credible_ball, Ball)
    assert isinstance(credible_ball.domain, MassWeightedHilbertSpace)
    assert_allclose(credible_ball.radius, radius, rtol=1e-12, atol=1e-12)

    boundary_point = mean + np.array([2.0 * radius, 0.0])
    outside_point = mean + np.array([2.0 * radius * 1.01, 0.0])

    assert credible_ball.is_element(boundary_point, rtol=1e-10)
    assert not credible_ball.is_element(outside_point, rtol=1e-10)


def test_credible_set_accepts_custom_rank():
    """A custom rank changes the chi-square degrees of freedom."""
    space = EuclideanSpace(5)
    measure = GaussianMeasure.from_standard_deviation(space, 1.0)
    probability = 0.95

    credible_set = measure.credible_set(probability, rank=3)

    expected_radius = np.sqrt(chi2.ppf(probability, df=3))
    assert_allclose(credible_set.radius, expected_radius, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("probability", [-0.1, 0.0, 1.0, 1.1])
def test_credible_set_rejects_invalid_probability(probability):
    """Credible probabilities must lie strictly between zero and one."""
    space = EuclideanSpace(2)
    measure = GaussianMeasure.from_standard_deviation(space, 1.0)

    with pytest.raises(ValueError, match="Probability"):
        measure.credible_set(probability)


@pytest.mark.parametrize("rank", [0, -1, 1.5])
def test_credible_set_rejects_invalid_rank(rank):
    """The chi-square rank must be a positive integer."""
    space = EuclideanSpace(2)
    measure = GaussianMeasure.from_standard_deviation(space, 1.0)

    with pytest.raises(ValueError, match="Rank"):
        measure.credible_set(0.8, rank=rank)


def test_credible_set_requires_rank_for_zero_dimensional_domain():
    """Basis-free spaces must provide an effective chi-square rank."""
    space = ZeroDimensionalSpace()
    measure = GaussianMeasure(
        covariance=space.identity_operator(),
        inverse_covariance=space.identity_operator(),
    )

    with pytest.raises(ValueError, match="Rank must be provided"):
        measure.credible_set(0.8)


def test_credible_set_rejects_invalid_geometry():
    """Only ellipsoid and Cameron-Martin ball geometries are supported."""
    space = EuclideanSpace(2)
    measure = GaussianMeasure.from_standard_deviation(space, 1.0)

    with pytest.raises(ValueError, match="Geometry"):
        measure.credible_set(0.8, geometry="cube")


def test_credible_set_requires_precision_operator():
    """A Mahalanobis credible set requires an available inverse covariance."""
    space = EuclideanSpace(2)
    covariance = space.identity_operator()
    measure = GaussianMeasure(covariance=covariance)

    with pytest.raises(AttributeError, match="Inverse covariance"):
        measure.credible_set(0.8)
