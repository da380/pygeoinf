"""
Tests for HyperPlane, HalfSpace, and PolyhedralSet geometric primitives.

These classes implement Phase 7 of the dual_master_implementation plan:
- HyperPlane: {x | ⟨a,x⟩ = b}
- HalfSpace: {x | ⟨a,x⟩ ≤ b} or {x | ⟨a,x⟩ ≥ b}
- PolyhedralSet: intersection of half-spaces
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.subsets import HyperPlane, HalfSpace, PolyhedralSet


@pytest.fixture
def space_2d():
    """2D Euclidean space for testing."""
    return EuclideanSpace(2)


@pytest.fixture
def space_3d():
    """3D Euclidean space for testing."""
    return EuclideanSpace(3)


# ============================================================================
# HyperPlane Tests
# ============================================================================

class TestHyperPlane:
    """Test suite for HyperPlane class."""

    def test_initialization_valid(self, space_2d):
        """Test valid initialization of hyperplane."""
        normal = np.array([1.0, 0.0])
        offset = 2.0
        plane = HyperPlane(space_2d, normal, offset)

        assert_allclose(plane.normal_vector, normal)
        assert plane.offset == offset
        assert_allclose(plane.normal_norm, 1.0)

    def test_initialization_zero_normal_raises(self, space_2d):
        """Test that zero normal vector raises error."""
        with pytest.raises((ValueError, AssertionError)):
            HyperPlane(space_2d, np.array([0.0, 0.0]), 1.0)

    def test_is_element_on_plane(self, space_2d):
        """Test membership for points on the hyperplane."""
        # Line: x + y = 1
        plane = HyperPlane(space_2d, np.array([1.0, 1.0]), 1.0)

        # Points on the line
        assert plane.is_element(np.array([0.5, 0.5]))
        assert plane.is_element(np.array([1.0, 0.0]))
        assert plane.is_element(np.array([0.0, 1.0]))
        assert plane.is_element(np.array([2.0, -1.0]))

    def test_is_element_off_plane(self, space_2d):
        """Test membership for points off the hyperplane."""
        plane = HyperPlane(space_2d, np.array([1.0, 0.0]), 2.0)

        # Points off the plane x = 2
        assert not plane.is_element(np.array([0.0, 0.0]))
        assert not plane.is_element(np.array([3.0, 5.0]))
        assert not plane.is_element(np.array([1.9, 0.0]))

    def test_is_element_tolerance(self, space_2d):
        """Test tolerance in membership checking."""
        plane = HyperPlane(space_2d, np.array([1.0, 0.0]), 10.0)

        # Point close to plane should be accepted with tolerance
        assert plane.is_element(np.array([10.05, 0.0]), rtol=1e-2)
        assert not plane.is_element(np.array([10.2, 0.0]), rtol=1e-2)

    def test_project_onto_plane(self, space_2d):
        """Test orthogonal projection onto hyperplane."""
        # Plane: y = 0 (x-axis)
        plane = HyperPlane(space_2d, np.array([0.0, 1.0]), 0.0)

        # Project point (3, 5) -> (3, 0)
        p = np.array([3.0, 5.0])
        proj = plane.project(p)
        assert_allclose(proj, np.array([3.0, 0.0]))

    def test_project_idempotence(self, space_2d):
        """Test that projecting twice gives same result."""
        plane = HyperPlane(space_2d, np.array([1.0, 1.0]), 5.0)

        p = np.array([10.0, -3.0])
        proj1 = plane.project(p)
        proj2 = plane.project(proj1)

        assert_allclose(proj1, proj2, atol=1e-12)

    def test_project_on_plane_unchanged(self, space_2d):
        """Test that points on plane are unchanged by projection."""
        plane = HyperPlane(space_2d, np.array([3.0, 4.0]), 10.0)

        # Find a point on the plane
        p_on = np.array([2.0, 1.0])  # 3*2 + 4*1 = 10 ✓
        proj = plane.project(p_on)

        assert_allclose(proj, p_on, atol=1e-12)

    def test_distance_perpendicular(self, space_2d):
        """Test perpendicular distance calculation."""
        # Plane: x = 5
        plane = HyperPlane(space_2d, np.array([1.0, 0.0]), 5.0)

        # Distance from (2, 10) to plane is |2 - 5| = 3
        dist = plane.distance_to(np.array([2.0, 10.0]))
        assert_allclose(dist, 3.0)

    def test_distance_zero_on_plane(self, space_2d):
        """Test distance is zero for points on plane."""
        plane = HyperPlane(space_2d, np.array([1.0, 1.0]), 0.0)

        p_on = np.array([5.0, -5.0])  # 5 + (-5) = 0 ✓
        assert plane.is_element(p_on)
        assert_allclose(plane.distance_to(p_on), 0.0, atol=1e-12)

    def test_boundary_is_self(self, space_2d):
        """Test that boundary of hyperplane is itself."""
        plane = HyperPlane(space_2d, np.array([1.0, 0.0]), 3.0)
        boundary = plane.boundary

        assert isinstance(boundary, HyperPlane)
        # Boundary should be the same hyperplane
        assert_allclose(boundary.normal_vector, plane.normal_vector)
        assert boundary.offset == plane.offset

    def test_normal_norm_property(self, space_3d):
        """Test that normal_norm is correctly computed."""
        normal = np.array([3.0, 4.0, 12.0])
        plane = HyperPlane(space_3d, normal, 0.0)

        expected_norm = np.linalg.norm(normal)
        assert_allclose(plane.normal_norm, expected_norm)


# ============================================================================
# HalfSpace Tests
# ============================================================================

class TestHalfSpace:
    """Test suite for HalfSpace class."""

    def test_initialization_inequality_types(self, space_2d):
        """Test initialization with different inequality types."""
        normal = np.array([1.0, 0.0])

        hs_leq = HalfSpace(space_2d, normal, 1.0, inequality_type='<=')
        hs_geq = HalfSpace(space_2d, normal, 1.0, inequality_type='>=')

        assert hs_leq.inequality_type == '<='
        assert hs_geq.inequality_type == '>='

    def test_initialization_zero_normal_raises(self, space_2d):
        """Test that zero normal vector raises error."""
        with pytest.raises((ValueError, AssertionError)):
            HalfSpace(space_2d, np.array([0.0, 0.0]), 1.0)

    def test_is_element_inside_leq(self, space_2d):
        """Test membership for points inside half-space (<=)."""
        # Half-space: x ≤ 5
        hs = HalfSpace(space_2d, np.array([1.0, 0.0]), 5.0, inequality_type='<=')

        assert hs.is_element(np.array([0.0, 0.0]))
        assert hs.is_element(np.array([3.0, 100.0]))
        assert hs.is_element(np.array([-10.0, 50.0]))

    def test_is_element_outside_leq(self, space_2d):
        """Test membership for points outside half-space (<=)."""
        hs = HalfSpace(space_2d, np.array([1.0, 0.0]), 5.0, inequality_type='<=')

        assert not hs.is_element(np.array([6.0, 0.0]))
        assert not hs.is_element(np.array([10.0, -5.0]))

    def test_is_element_boundary(self, space_2d):
        """Test membership for points on boundary."""
        hs = HalfSpace(space_2d, np.array([1.0, 1.0]), 10.0, inequality_type='<=')

        # Boundary: x + y = 10
        assert hs.is_element(np.array([5.0, 5.0]))
        assert hs.is_element(np.array([0.0, 10.0]))
        assert hs.is_element(np.array([10.0, 0.0]))

    def test_inequality_type_comparison(self, space_2d):
        """Test that <= and >= behave oppositely."""
        normal = np.array([1.0, 0.0])

        hs_leq = HalfSpace(space_2d, normal, 0.0, inequality_type='<=')
        hs_geq = HalfSpace(space_2d, normal, 0.0, inequality_type='>=')

        p_neg = np.array([-5.0, 0.0])  # x < 0
        p_pos = np.array([5.0, 0.0])   # x > 0

        assert hs_leq.is_element(p_neg) and not hs_leq.is_element(p_pos)
        assert hs_geq.is_element(p_pos) and not hs_geq.is_element(p_neg)

    def test_support_function_lazy_init(self, space_2d):
        """Test that support function is lazily initialized."""
        hs = HalfSpace(space_2d, np.array([1.0, 0.0]), 1.0)

        # Access support_function property
        sf = hs.support_function

        from pygeoinf.convex_analysis import HalfSpaceSupportFunction
        assert isinstance(sf, HalfSpaceSupportFunction)

    def test_support_function_bounded_direction_leq(self, space_2d):
        """Test support function for bounded direction (<=)."""
        # Half-space: x ≤ 10 (normal = [1, 0], offset = 10)
        hs = HalfSpace(space_2d, np.array([1.0, 0.0]), 10.0, inequality_type='<=')
        sf = hs.support_function

        # For <=: σ(q) = α·b when q = α·a with α ≥ 0, else +∞
        q_pos = np.array([2.0, 0.0])  # α = 2 ≥ 0, should be bounded
        q_neg = np.array([-1.0, 0.0])  # α = -1 < 0, should be unbounded

        result_pos = sf(q_pos)
        result_neg = sf(q_neg)

        assert np.isfinite(result_pos), "Expected finite value for α ≥ 0"
        assert_allclose(result_pos, 2.0 * 10.0, rtol=1e-10)  # α * b = 2 * 10 = 20
        assert np.isinf(result_neg) and result_neg > 0, "Expected +∞ for α < 0"

    def test_support_function_bounded_direction_geq(self, space_2d):
        """Test support function for bounded direction (>=)."""
        # Half-space: x ≥ 10 (normal = [1, 0], offset = 10)
        hs = HalfSpace(space_2d, np.array([1.0, 0.0]), 10.0, inequality_type='>=')
        sf = hs.support_function

        # For >=: σ(q) = α·b when q = α·a with α ≤ 0, else +∞
        q_pos = np.array([3.0, 0.0])  # α = 3 > 0, should be unbounded
        q_neg = np.array([-2.0, 0.0])  # α = -2 ≤ 0, should be bounded

        result_pos = sf(q_pos)
        result_neg = sf(q_neg)

        assert np.isinf(result_pos) and result_pos > 0, "Expected +∞ for α > 0"
        assert np.isfinite(result_neg), "Expected finite value for α ≤ 0"
        assert_allclose(result_neg, -2.0 * 10.0, rtol=1e-10)  # α * b = -2 * 10 = -20

    def test_support_function_perpendicular_unbounded(self, space_2d):
        """Test that perpendicular directions are unbounded."""
        # Half-space: x ≤ 5
        hs = HalfSpace(space_2d, np.array([1.0, 0.0]), 5.0, inequality_type='<=')
        sf = hs.support_function

        # Query perpendicular to normal: q = [0, 1]
        q_perp = np.array([0.0, 1.0])
        result = sf(q_perp)

        assert np.isinf(result) and result > 0, "Perpendicular direction should be unbounded"

    def test_support_point_minimum_norm(self, space_2d):
        """Test that support_point returns minimum-norm boundary point."""
        # Half-space: x + y ≤ 10
        hs = HalfSpace(space_2d, np.array([1.0, 1.0]), 10.0, inequality_type='<=')
        sf = hs.support_function

        # Bounded query: q = [1, 1] (parallel to normal, α = 1 ≥ 0)
        q = np.array([1.0, 1.0])
        sp = sf.support_point(q)

        # Should return minimum-norm boundary point
        assert sp is not None, "support_point should return a point for finite σ"

        # Should lie on boundary: x + y = 10
        assert_allclose(np.dot(np.array([1.0, 1.0]), sp), 10.0, atol=1e-10)

        # Minimum norm point: x_min = (b/||a||²) a = (10/2) [1,1] = [5,5]
        assert_allclose(sp, np.array([5.0, 5.0]), atol=1e-10)

    def test_project_onto_boundary(self, space_2d):
        """Test projection returns points on boundary hyperplane."""
        hs = HalfSpace(space_2d, np.array([0.0, 1.0]), 3.0, inequality_type='<=')

        # Project point onto boundary y = 3
        p = np.array([7.0, 10.0])
        proj = hs.project(p)

        assert_allclose(proj, np.array([7.0, 3.0]))

    def test_boundary_returns_hyperplane(self, space_2d):
        """Test that boundary property returns a HyperPlane."""
        hs = HalfSpace(space_2d, np.array([1.0, 1.0]), 5.0)
        boundary = hs.boundary

        assert isinstance(boundary, HyperPlane)
        assert_allclose(boundary.normal_vector, hs.normal_vector)
        assert boundary.offset == hs.offset

    def test_is_unbounded(self, space_2d):
        """Test that half-spaces are always unbounded."""
        hs = HalfSpace(space_2d, np.array([1.0, 0.0]), 5.0)

        assert not hs.is_bounded(), "Half-spaces should always be unbounded"


# ============================================================================
# PolyhedralSet Tests
# ============================================================================

class TestPolyhedralSet:
    """Test suite for PolyhedralSet (intersection of half-spaces)."""

    def test_initialization_valid(self, space_2d):
        """Test valid initialization of polyhedral set."""
        hs1 = HalfSpace(space_2d, np.array([1.0, 0.0]), 5.0, inequality_type='<=')
        hs2 = HalfSpace(space_2d, np.array([0.0, 1.0]), 5.0, inequality_type='<=')

        poly = PolyhedralSet(space_2d, [hs1, hs2])

        assert len(poly.half_spaces) == 2

    def test_initialization_mismatched_domains_raises(self, space_2d, space_3d):
        """Test that mismatched domains raise error."""
        hs1 = HalfSpace(space_2d, np.array([1.0, 0.0]), 1.0)
        hs2 = HalfSpace(space_3d, np.array([1.0, 0.0, 0.0]), 1.0)

        with pytest.raises((ValueError, AssertionError)):
            PolyhedralSet(space_2d, [hs1, hs2])

    def test_is_element_satisfies_all_constraints(self, space_2d):
        """Test that membership requires satisfying all constraints."""
        # Box: 0 ≤ x ≤ 10, 0 ≤ y ≤ 10
        hs1 = HalfSpace(space_2d, np.array([1.0, 0.0]), 10.0, inequality_type='<=')
        hs2 = HalfSpace(space_2d, np.array([-1.0, 0.0]), 0.0, inequality_type='<=')  # x ≥ 0
        hs3 = HalfSpace(space_2d, np.array([0.0, 1.0]), 10.0, inequality_type='<=')
        hs4 = HalfSpace(space_2d, np.array([0.0, -1.0]), 0.0, inequality_type='<=')  # y ≥ 0

        poly = PolyhedralSet(space_2d, [hs1, hs2, hs3, hs4])

        # Inside box
        assert poly.is_element(np.array([5.0, 5.0]))
        assert poly.is_element(np.array([0.0, 0.0]))
        assert poly.is_element(np.array([10.0, 10.0]))

        # Outside box
        assert not poly.is_element(np.array([11.0, 5.0]))
        assert not poly.is_element(np.array([5.0, -1.0]))

    def test_is_element_simplex(self, space_3d):
        """Test standard simplex: x ≥ 0, y ≥ 0, z ≥ 0, x+y+z ≤ 1."""
        # x ≥ 0: -x ≤ 0
        hs_x = HalfSpace(space_3d, np.array([-1.0, 0.0, 0.0]), 0.0, inequality_type='<=')
        # y ≥ 0: -y ≤ 0
        hs_y = HalfSpace(space_3d, np.array([0.0, -1.0, 0.0]), 0.0, inequality_type='<=')
        # z ≥ 0: -z ≤ 0
        hs_z = HalfSpace(space_3d, np.array([0.0, 0.0, -1.0]), 0.0, inequality_type='<=')
        # x + y + z ≤ 1
        hs_sum = HalfSpace(space_3d, np.array([1.0, 1.0, 1.0]), 1.0, inequality_type='<=')

        simplex = PolyhedralSet(space_3d, [hs_x, hs_y, hs_z, hs_sum])

        # Inside simplex
        assert simplex.is_element(np.array([0.25, 0.25, 0.25]))
        assert simplex.is_element(np.array([0.0, 0.0, 0.0]))
        assert simplex.is_element(np.array([1.0, 0.0, 0.0]))

        # Outside simplex
        assert not simplex.is_element(np.array([0.5, 0.5, 0.5]))  # sum > 1
        assert not simplex.is_element(np.array([-0.1, 0.5, 0.5]))  # x < 0

    def test_half_spaces_property(self, space_2d):
        """Test that half_spaces property is accessible."""
        hs1 = HalfSpace(space_2d, np.array([1.0, 0.0]), 1.0)
        hs2 = HalfSpace(space_2d, np.array([0.0, 1.0]), 1.0)

        poly = PolyhedralSet(space_2d, [hs1, hs2])

        assert len(poly.half_spaces) == 2
        assert poly.half_spaces[0] is hs1
        assert poly.half_spaces[1] is hs2

    def test_support_function_returns_none(self, space_2d):
        """Test that support function returns None (requires LP solver)."""
        hs = HalfSpace(space_2d, np.array([1.0, 0.0]), 1.0)
        poly = PolyhedralSet(space_2d, [hs])

        assert poly.support_function is None, "PolyhedralSet support function requires LP solver"


# ============================================================================
# Numerical Robustness Tests
# ============================================================================

class TestNumericalRobustness:
    """Test numerical edge cases and tolerance handling."""

    def test_nearly_parallel_directions(self, space_2d):
        """Test support function with nearly parallel query directions."""
        hs = HalfSpace(space_2d, np.array([1.0, 0.0]), 10.0, inequality_type='<=')
        sf = hs.support_function

        # Query almost parallel but slightly off: q = [-1, 1e-10]
        q_almost = np.array([-1.0, 1e-10])
        result = sf(q_almost)

        # With default tolerance, should be unbounded (perpendicular component detected)
        assert np.isinf(result) and result > 0

    def test_large_offset_values(self, space_2d):
        """Test with large offset values."""
        large_offset = 1e10
        plane = HyperPlane(space_2d, np.array([1.0, 0.0]), large_offset)

        p_on = np.array([large_offset, 0.0])
        assert plane.is_element(p_on)
        assert_allclose(plane.distance_to(p_on), 0.0, atol=1e-6)

    def test_small_normal_vectors(self, space_2d):
        """Test with small but non-zero normal vectors."""
        small_normal = np.array([1e-8, 0.0])
        plane = HyperPlane(space_2d, small_normal, 1.0)

        # Normal norm should be 1e-8
        assert_allclose(plane.normal_norm, 1e-8, rtol=1e-10)

