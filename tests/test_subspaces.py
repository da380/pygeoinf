"""
Tests for the subspaces module.
"""

import numpy as np
import pytest

from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.subsets import EmptySet
from pygeoinf.subspaces import (
    AffineSubspace,
    OrthogonalProjector,
)


@pytest.fixture
def space_3d():
    """Returns a 3D Euclidean space."""
    return EuclideanSpace(3)


class TestOrthogonalProjector:
    def test_from_basis(self, space_3d):
        # Basis for xy-plane
        e1 = space_3d.from_components(np.array([1.0, 0.0, 0.0]))
        e2 = space_3d.from_components(np.array([0.0, 1.0, 0.0]))

        P = OrthogonalProjector.from_basis(space_3d, [e1, e2])

        # Project vector (1, 2, 3) -> (1, 2, 0)
        v = space_3d.from_components(np.array([1.0, 2.0, 3.0]))
        proj = P(v)
        expected = space_3d.from_components(np.array([1.0, 2.0, 0.0]))

        assert np.allclose(proj, expected)

        # Complement should map to (0, 0, 3)
        comp = P.complement(v)
        expected_comp = space_3d.from_components(np.array([0.0, 0.0, 3.0]))
        assert np.allclose(comp, expected_comp)


@pytest.mark.filterwarnings("ignore:Constructing a subspace")
class TestAffineSubspace:
    def test_construction_and_projection(self, space_3d):
        # Line parallel to z-axis passing through (1, 1, 0)
        # Tangent space basis: e3
        e3 = space_3d.from_components(np.array([0.0, 0.0, 1.0]))
        x0 = space_3d.from_components(np.array([1.0, 1.0, 0.0]))

        subspace = AffineSubspace.from_tangent_basis(space_3d, [e3], translation=x0)

        # Project origin (0,0,0) -> (1,1,0)
        zero = space_3d.zero
        proj = subspace.project(zero)

        assert np.allclose(proj, x0)

        # Project (2, 2, 5) -> (1, 1, 5)
        # P_A(x) = P(x-x0) + x0
        # x-x0 = (1, 1, 5). P(x-x0) = (0, 0, 5). Result = (1, 1, 5)
        v = space_3d.from_components(np.array([2.0, 2.0, 5.0]))
        proj_v = subspace.project(v)
        expected = space_3d.from_components(np.array([1.0, 1.0, 5.0]))

        assert np.allclose(proj_v, expected)

    def test_is_element(self, space_3d):
        # xy-plane at z=1
        e1 = space_3d.from_components(np.array([1.0, 0.0, 0.0]))
        e2 = space_3d.from_components(np.array([0.0, 1.0, 0.0]))
        x0 = space_3d.from_components(np.array([0.0, 0.0, 1.0]))

        subspace = AffineSubspace.from_tangent_basis(space_3d, [e1, e2], translation=x0)

        p_in = space_3d.from_components(np.array([5.0, -3.0, 1.0]))
        assert subspace.is_element(p_in)

        p_out = space_3d.from_components(np.array([5.0, -3.0, 1.1]))
        assert not subspace.is_element(p_out)

    def test_boundary(self, space_3d):
        # Affine subspaces are closed manifolds, boundary is empty
        e1 = space_3d.from_components(np.array([1.0, 0.0, 0.0]))
        subspace = AffineSubspace.from_tangent_basis(space_3d, [e1])
        assert isinstance(subspace.boundary, EmptySet)
