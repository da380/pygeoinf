"""
Tests for the subspaces module, covering OrthogonalProjector, LinearSubspace,
and AffineSubspace.
"""

import pytest
import numpy as np

from pygeoinf.subspaces import AffineSubspace, LinearSubspace, OrthogonalProjector
from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.symmetric_space.circle import Lebesgue
from pygeoinf.linear_operators import LinearOperator
from pygeoinf.linear_solvers import CholeskySolver

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def r3():
    return EuclideanSpace(3)


@pytest.fixture
def circle_space():
    return Lebesgue(8, radius=1.0)


# =============================================================================
# OrthogonalProjector Tests
# =============================================================================


def test_euclidean_subspace_basis(r3):
    v1 = r3.from_components(np.array([1.0, 0.0, 0.0]))
    v2 = r3.from_components(np.array([0.0, 1.0, 0.0]))
    P = OrthogonalProjector.from_basis(r3, [v1, v2], orthonormalize=True)

    P.check(n_checks=10)
    assert np.allclose(
        P(r3.from_components(np.array([2.0, 3.0, 0.0]))), [2.0, 3.0, 0.0]
    )
    assert np.allclose(
        P(r3.from_components(np.array([0.0, 0.0, 5.0]))), [0.0, 0.0, 0.0]
    )


def test_circle_low_freq_subspace(circle_space):
    basis = [
        circle_space.project_function(lambda t: 1.0),
        circle_space.project_function(lambda t: np.cos(t)),
        circle_space.project_function(lambda t: np.sin(t)),
    ]
    P = OrthogonalProjector.from_basis(circle_space, basis, orthonormalize=True)
    assert np.allclose(P(basis[1]), basis[1])

    high_freq = circle_space.project_function(lambda t: np.cos(2 * t))
    assert circle_space.norm(P(high_freq)) < 1e-12


# =============================================================================
# LinearSubspace Tests
# =============================================================================


def test_linear_subspace_r3(r3):
    e1 = r3.basis_vector(0)
    subspace = LinearSubspace.from_basis(r3, [e1])
    assert subspace.is_element(r3.multiply(2.0, e1))
    assert not subspace.is_element(r3.from_components(np.array([2.0, 1.0, 0.0])))


def test_linear_subspace_from_kernel(r3):
    """
    Test defining subspace as kernel. implicit solver usage (Cholesky).
    """
    codomain = EuclideanSpace(1)
    C_mat = np.array([[1.0, 1.0, 1.0]])
    C = LinearOperator.from_matrix(r3, codomain, C_mat, galerkin=True)

    # Uses default CholeskySolver automatically
    subspace = LinearSubspace.from_kernel(C)

    assert subspace.is_element(r3.from_components(np.array([1.0, -1.0, 0.0])))
    assert not subspace.is_element(r3.from_components(np.array([1.0, 1.0, 1.0])))

    # Check property name update
    assert subspace.has_explicit_equation


# =============================================================================
# AffineSubspace Tests
# =============================================================================


def test_affine_subspace_projection_r3(r3):
    e1 = r3.basis_vector(0)
    e2 = r3.basis_vector(1)
    P_linear = OrthogonalProjector.from_basis(r3, [e1, e2])
    z_offset = r3.from_components(np.array([0.0, 0.0, 2.0]))
    affine_plane = AffineSubspace(P_linear, translation=z_offset)

    p_in = r3.from_components(np.array([5.0, 5.0, 10.0]))
    p_expected = r3.from_components(np.array([5.0, 5.0, 2.0]))
    assert np.allclose(affine_plane.project(p_in), p_expected)


def test_linear_equation_factory_r3(r3):
    """
    Test constructing from B(u)=w using explicit solver passing.
    """
    codomain = EuclideanSpace(1)
    e_z = r3.basis_vector(2)
    B = LinearOperator(
        r3,
        codomain,
        lambda u: np.array([r3.inner_product(u, e_z)]),
        adjoint_mapping=lambda w: r3.multiply(w[0], e_z),
    )
    w = codomain.from_components(np.array([2.0]))

    # Explicitly pass solver (positional or keyword)
    subspace = AffineSubspace.from_linear_equation(B, w, CholeskySolver(galerkin=True))

    expected_trans = r3.from_components(np.array([0.0, 0.0, 2.0]))
    assert np.allclose(subspace.translation, expected_trans)


def test_affine_constraint_function_space(circle_space):
    codomain = EuclideanSpace(1)
    ones_func = circle_space.project_function(lambda t: 1.0)
    C = LinearOperator(
        circle_space,
        codomain,
        lambda f: np.array([circle_space.inner_product(f, ones_func)]),
        adjoint_mapping=lambda y: circle_space.multiply(y[0], ones_func),
    )

    target_mean = 5.0
    norm_sq = circle_space.squared_norm(ones_func)
    translation = circle_space.multiply(target_mean / norm_sq, ones_func)

    linear_subspace = LinearSubspace.from_kernel(C)
    affine_subspace = AffineSubspace(linear_subspace.projector, translation=translation)

    sine = circle_space.project_function(lambda t: np.sin(t))
    proj_sine = affine_subspace.project(sine)
    expected = circle_space.add(translation, sine)
    assert np.allclose(proj_sine, expected)


def test_affine_from_complement_basis_generates_constraints(r3):
    e_z = r3.basis_vector(2)
    translation = r3.from_components(np.array([0.0, 0.0, 2.0]))

    # This factory generates B implicitly. It defaults to Identity solver internally.
    subspace = AffineSubspace.from_complement_basis(r3, [e_z], translation=translation)

    assert subspace.has_explicit_equation
    # Check that we can condition using the implicit solver (consistency check)
    assert subspace._solver is not None


def test_tangent_basis_implicit_constraint(r3):
    """
    Test that a subspace defined only by tangent vectors (implicit)
    can still expose a constraint operator (I-P) and accept a solver.
    """
    # Define subspace: XY plane (z=0) via tangent basis
    e1 = r3.basis_vector(0)
    e2 = r3.basis_vector(1)

    # We pass a solver explicitly to allow this implicit subspace to be used
    # for Bayesian conditioning later if needed.
    subspace = AffineSubspace.from_tangent_basis(
        r3, [e1, e2], solver=CholeskySolver(galerkin=True)
    )

    # 1. It should NOT have an explicit B(u)=w equation
    assert not subspace.has_explicit_equation

    # 2. But it SHOULD provide a constraint operator B = I - P
    B_implicit = subspace.constraint_operator
    assert B_implicit is not None

    # Test B on a vector in the subspace (should be zero)
    v_in = r3.from_components(np.array([1.0, 2.0, 0.0]))
    # B(v) = (I-P)v = v - v = 0
    assert np.allclose(B_implicit(v_in), np.zeros(3))

    # Test B on a vector orthogonal (should be itself)
    v_perp = r3.from_components(np.array([0.0, 0.0, 5.0]))
    assert np.allclose(B_implicit(v_perp), v_perp)

    # 3. Check solver is attached
    assert subspace._solver is not None
