"""
Tests for the subset module.
"""

import numpy as np
import pytest

from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.linear_operators import DiagonalSparseMatrixLinearOperator
from pygeoinf.subset import (
    EmptySet,
    Ball,
    Sphere,
    Ellipsoid,
    HalfSpace,
    Hyperplane,
    NormalisedEllipsoid,
)


@pytest.fixture
def space_2d():
    """Returns a 2D Euclidean space for geometric tests."""
    return EuclideanSpace(2)


@pytest.fixture
def space_3d():
    """Returns a 3D Euclidean space."""
    return EuclideanSpace(3)


class TestEmptySet:
    def test_initialization(self, space_2d):
        # Can init with or without domain
        e1 = EmptySet()
        assert e1.is_empty

        e2 = EmptySet(space_2d)
        assert e2.domain == space_2d
        assert e2.is_empty

    def test_behavior(self, space_2d):
        e = EmptySet(space_2d)
        x = space_2d.random()

        assert not e.is_element(x)
        assert e.boundary.is_empty
        assert isinstance(e.boundary, EmptySet)


class TestBallAndSphere:
    def test_ball_membership(self, space_2d):
        center = space_2d.from_components(np.array([0.0, 0.0]))
        ball = Ball(space_2d, center, radius=1.0)

        # Inside
        p_in = space_2d.from_components(np.array([0.5, 0.0]))
        assert ball.is_element(p_in)

        # Outside
        p_out = space_2d.from_components(np.array([1.5, 0.0]))
        assert not ball.is_element(p_out)

        # Boundary (closed ball includes it)
        p_bound = space_2d.from_components(np.array([1.0, 0.0]))
        assert ball.is_element(p_bound)

    def test_open_ball(self, space_2d):
        center = space_2d.zero
        ball = Ball(space_2d, center, radius=1.0, open_set=True)

        # Boundary point should NOT be in open ball mathematically.
        # We must set rtol=0.0 to test strict inequality, otherwise
        # the tolerance margin will include the boundary.
        p_bound = space_2d.from_components(np.array([1.0, 0.0]))
        assert not ball.is_element(p_bound, rtol=0.0)

        # But should be allowed with sufficient tolerance (default behavior)
        assert ball.is_element(p_bound, rtol=1e-5)

    def test_sphere_boundary(self, space_2d):
        center = space_2d.zero
        ball = Ball(space_2d, center, radius=1.0)
        sphere = ball.boundary

        assert isinstance(sphere, Sphere)

        # Use numpy comparison for array-like vectors
        assert np.allclose(sphere.center, center)
        assert sphere.radius == 1.0
        assert sphere.boundary.is_empty

        # Sphere membership
        p_bound = space_2d.from_components(np.array([1.0, 0.0]))
        p_in = space_2d.from_components(np.array([0.5, 0.0]))

        assert sphere.is_element(p_bound)
        assert not sphere.is_element(p_in)

    def test_convexity_check(self, space_2d):
        # The Ball class inherits check() from ConvexSubset
        ball = Ball(space_2d, space_2d.zero, radius=1.0)
        # Should pass without assertion error
        ball.check(20)


class TestEllipsoid:
    def test_ellipsoid_geometry(self, space_2d):
        """Test an ellipse stretched along x-axis: 0.25*x^2 + y^2 <= 1"""
        # Matrix A = diag([0.25, 1.0])
        # This implies radii: r_x = 2, r_y = 1
        diag_vals = np.array([0.25, 1.0])
        A = DiagonalSparseMatrixLinearOperator.from_diagonal_values(
            space_2d, space_2d, diag_vals
        )

        center = space_2d.zero
        ellipsoid = Ellipsoid(space_2d, center, radius=1.0, operator=A)

        # Check point (1.5, 0) -> 0.25*(1.5)^2 = 0.5625 <= 1 (True)
        p1 = space_2d.from_components(np.array([1.5, 0.0]))
        assert ellipsoid.is_element(p1)

        # Check point (0, 0.9) -> 0.9^2 = 0.81 <= 1 (True)
        p2 = space_2d.from_components(np.array([0.0, 0.9]))
        assert ellipsoid.is_element(p2)

        # Check point (2.1, 0) -> 0.25*(2.1)^2 = 1.1025 > 1 (False)
        p3 = space_2d.from_components(np.array([2.1, 0.0]))
        assert not ellipsoid.is_element(p3)

        # Convexity check
        ellipsoid.check(20)

    def test_normalization(self, space_2d):
        # Create ellipsoid with r=2 and A=I
        # Condition: ||x||^2 <= 4
        A = space_2d.identity_operator()
        ellipsoid = Ellipsoid(space_2d, space_2d.zero, radius=2.0, operator=A)

        norm_ellipsoid = ellipsoid.normalized

        assert isinstance(norm_ellipsoid, NormalisedEllipsoid)
        assert norm_ellipsoid.radius == 1.0

        # The operator should be scaled by 1/r^2 = 1/4
        # New condition: <0.25*I x, x> <= 1  <==> 0.25*||x||^2 <= 1 <==> ||x||^2 <= 4
        # So they represent the same set.

        p = space_2d.from_components(np.array([1.9, 0.0]))
        assert ellipsoid.is_element(p)
        assert norm_ellipsoid.is_element(p)


class TestHalfSpace:
    def test_halfspace_from_vector(self, space_2d):
        # Define halfspace x >= 1 (normal vector n=[-1, 0], level <= -1)
        # Or more simply: <n, x> <= c
        # Let n = [1, 0], c = 1. So x <= 1.

        n = space_2d.from_components(np.array([1.0, 0.0]))
        hs = HalfSpace.from_vector(space_2d, n, level=1.0)

        # (0.5, 0) -> <n,x>=0.5 <= 1 (True)
        assert hs.is_element(space_2d.from_components(np.array([0.5, 0.0])))

        # (1.5, 0) -> <n,x>=1.5 > 1 (False)
        assert not hs.is_element(space_2d.from_components(np.array([1.5, 0.0])))

        # Boundary is hyperplane x = 1
        hp = hs.boundary
        assert isinstance(hp, Hyperplane)
        assert hp.is_element(
            space_2d.from_components(np.array([1.0, 5.0]))
        )  # y can be anything

    def test_convexity(self, space_3d):
        n = space_3d.random()
        hs = HalfSpace.from_vector(space_3d, n, level=0.5)
        hs.check(20)
