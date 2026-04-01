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
    def test_get_tangent_basis_axis_aligned_1d(self, space_3d):
        """Axis-aligned 1D subspace: basis should have dimension 1."""
        e3 = space_3d.from_components(np.array([0.0, 0.0, 1.0]))
        subspace = AffineSubspace.from_tangent_basis(space_3d, [e3])
        basis = subspace.get_tangent_basis()
        assert len(basis) == 1
        # Result should be parallel to e3 (up to sign)
        assert np.allclose(np.abs(basis[0]), [0.0, 0.0, 1.0])

    def test_get_tangent_basis_axis_aligned_2d(self, space_3d):
        """Axis-aligned 2D subspace (xy-plane): basis should have dimension 2."""
        e1 = space_3d.from_components(np.array([1.0, 0.0, 0.0]))
        e2 = space_3d.from_components(np.array([0.0, 1.0, 0.0]))
        subspace = AffineSubspace.from_tangent_basis(space_3d, [e1, e2])
        basis = subspace.get_tangent_basis()
        assert len(basis) == 2

    def test_get_tangent_basis_diagonal_1d(self, space_3d):
        """Diagonal 1D subspace (equal-components line): must report dimension 1, not 3."""
        diag = space_3d.from_components(np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0))
        subspace = AffineSubspace.from_tangent_basis(space_3d, [diag])
        basis = subspace.get_tangent_basis()
        assert len(basis) == 1, (
            f"Expected dimension 1 for diagonal 1D subspace, got {len(basis)}"
        )
        # The single basis vector should span the same line as diag.
        np.testing.assert_allclose(
            np.abs(np.dot(basis[0], diag)), 1.0, rtol=1e-9,
            err_msg="Basis vector should be parallel to the original diagonal direction"
        )

    def test_get_tangent_basis_diagonal_2d(self, space_3d):
        """Non-axis-aligned 2D subspace: must report dimension 2."""
        # Plane spanned by (1,1,0)/sqrt(2) and (0,0,1)
        v1 = space_3d.from_components(np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0))
        v2 = space_3d.from_components(np.array([0.0, 0.0, 1.0]))
        subspace = AffineSubspace.from_tangent_basis(space_3d, [v1, v2])
        basis = subspace.get_tangent_basis()
        assert len(basis) == 2, (
            f"Expected dimension 2 for oblique 2D subspace, got {len(basis)}"
        )
        # Basis vectors should be orthonormal.
        np.testing.assert_allclose(
            np.dot(basis[0], basis[0]), 1.0, rtol=1e-9
        )
        np.testing.assert_allclose(
            np.dot(basis[1], basis[1]), 1.0, rtol=1e-9
        )
        np.testing.assert_allclose(
            np.dot(basis[0], basis[1]), 0.0, atol=1e-9
        )

    def test_get_tangent_basis_oblique_1d_with_translation(self, space_3d):
        """Translation should not affect the tangent basis dimension."""
        diag = space_3d.from_components(np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0))
        translation = space_3d.from_components(np.array([3.0, 1.0, -2.0]))
        subspace = AffineSubspace.from_tangent_basis(space_3d, [diag], translation=translation)
        basis = subspace.get_tangent_basis()
        assert len(basis) == 1