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

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def r3():
    """Provides a 3D Euclidean space."""
    return EuclideanSpace(3)


@pytest.fixture
def circle_space():
    """Provides the Lebesgue space L2(S1) with max frequency 8."""
    return Lebesgue(8, radius=1.0)


# =============================================================================
# OrthogonalProjector Tests (Low-Level Engine)
# =============================================================================


def test_euclidean_subspace_basis(r3):
    """
    Test projection onto the XY plane in R^3 defined by basis vectors.
    """
    # Define basis for XY plane
    v1 = r3.from_components(np.array([1.0, 0.0, 0.0]))
    v2 = r3.from_components(np.array([0.0, 1.0, 0.0]))

    # Pass domain explicitly
    P = OrthogonalProjector.from_basis(r3, [v1, v2], orthonormalize=True)

    # 1. Check Axioms (Self-adjoint, Linear, etc.)
    P.check(n_checks=10)

    # 2. Check Action on Subspace (should be identity)
    x_in = r3.from_components(np.array([2.0, 3.0, 0.0]))
    assert np.allclose(P(x_in), x_in)

    # 3. Check Action on Complement (should be zero)
    z_axis = r3.from_components(np.array([0.0, 0.0, 5.0]))
    assert np.allclose(P(z_axis), np.zeros(3))

    # 4. Check General Vector
    v = r3.from_components(np.array([1.0, 2.0, 3.0]))
    proj = P(v)
    expected = r3.from_components(np.array([1.0, 2.0, 0.0]))
    assert np.allclose(proj, expected)


def test_circle_low_freq_subspace(circle_space):
    """
    Construct a projector onto the subspace spanned by {1, cos(theta), sin(theta)}.
    """
    # Create basis functions
    f0 = lambda t: 1.0
    f1 = lambda t: np.cos(t)
    f2 = lambda t: np.sin(t)

    basis = [
        circle_space.project_function(f0),
        circle_space.project_function(f1),
        circle_space.project_function(f2),
    ]

    # Pass domain explicitly
    P = OrthogonalProjector.from_basis(circle_space, basis, orthonormalize=True)

    # 1. Test Idempotence on a basis vector
    v_in = basis[1]  # cos(theta)
    assert np.allclose(P(v_in), v_in)

    # 2. Test Rejection of high frequencies
    # cos(2*theta) is orthogonal to {1, cos, sin} on the circle
    high_freq = circle_space.project_function(lambda t: np.cos(2 * t))

    # Should be projected to 0 (machine precision)
    proj = P(high_freq)
    assert circle_space.norm(proj) < 1e-12


# =============================================================================
# LinearSubspace Tests (High-Level API)
# =============================================================================


def test_linear_subspace_r3(r3):
    """
    Tests the LinearSubspace wrapper using the from_basis factory.
    """
    # Subspace: The X-axis
    e1 = r3.basis_vector(0)

    # Pass domain explicitly
    subspace = LinearSubspace.from_basis(r3, [e1])

    # 1. Test Membership
    # (2, 0, 0) is in the subspace
    assert subspace.is_element(r3.multiply(2.0, e1))

    # (2, 1, 0) is NOT in the subspace
    v_out = r3.from_components(np.array([2.0, 1.0, 0.0]))
    assert not subspace.is_element(v_out)

    # 2. Test Complement
    # Complement of X-axis is YZ-plane
    comp = subspace.complement
    e2 = r3.basis_vector(1)
    e3 = r3.basis_vector(2)

    assert comp.is_element(e2)
    assert comp.is_element(e3)
    assert not comp.is_element(e1)


def test_linear_subspace_from_kernel(r3):
    """
    Tests defining a LinearSubspace as the kernel of an operator.
    Constraint: x + y + z = 0.
    """
    codomain = EuclideanSpace(1)
    C_mat = np.array([[1.0, 1.0, 1.0]])
    C = LinearOperator.from_matrix(r3, codomain, C_mat, galerkin=True)

    subspace = LinearSubspace.from_kernel(C)

    # Vector (1, -1, 0) is in the kernel
    v_in = r3.from_components(np.array([1.0, -1.0, 0.0]))
    assert subspace.is_element(v_in)

    # Vector (1, 1, 1) is orthogonal to the kernel
    v_out = r3.from_components(np.array([1.0, 1.0, 1.0]))
    assert not subspace.is_element(v_out)

    # Projecting v_out should yield 0
    assert np.allclose(subspace.project(v_out), np.zeros(3))


# =============================================================================
# AffineSubspace Tests (High-Level API)
# =============================================================================


def test_affine_subspace_projection_r3(r3):
    """
    Tests projection onto an affine plane z = 2.
    """
    # 1. Define Linear Projector onto XY plane (z=0)
    e1 = r3.basis_vector(0)
    e2 = r3.basis_vector(1)

    # Pass domain explicitly
    P_linear = OrthogonalProjector.from_basis(r3, [e1, e2])

    # 2. Define Translation to z=2
    z_offset = r3.from_components(np.array([0.0, 0.0, 2.0]))

    # 3. Create Affine Subspace using constructor
    affine_plane = AffineSubspace(P_linear, translation=z_offset)

    # 4. Test Projection of Origin
    # Projecting (0,0,0) should yield (0,0,2) -- the closest point on the plane
    origin = r3.zero
    proj_origin = affine_plane.project(origin)
    assert np.allclose(proj_origin, z_offset)

    # 5. Test Projection of Arbitrary Point
    # Point (5, 5, 10) -> Should project to (5, 5, 2)
    p_in = r3.from_components(np.array([5.0, 5.0, 10.0]))
    p_expected = r3.from_components(np.array([5.0, 5.0, 2.0]))

    p_out = affine_plane.project(p_in)
    assert np.allclose(p_out, p_expected)
    assert affine_plane.is_element(p_out)


def test_linear_equation_factory_r3(r3):
    """
    Test constructing an affine subspace from B(u) = w.
    We use B(u) = u_z (projection onto Z axis).
    We set w = 2.
    Expected subspace: The plane z=2.
    """
    # 1. Define Operator B: R3 -> R1, B(u) = <u, e_z>
    codomain = EuclideanSpace(1)
    e_z = r3.basis_vector(2)  # [0, 0, 1]

    # Simple rank-1 operator
    def mapping(u):
        return np.array([r3.inner_product(u, e_z)])

    def adjoint(w_comp):
        return r3.multiply(w_comp[0], e_z)

    B = LinearOperator(r3, codomain, mapping, adjoint_mapping=adjoint)

    # 2. Define Target w = 2
    w = codomain.from_components(np.array([2.0]))

    # 3. Create Subspace using implicit factory
    subspace = AffineSubspace.from_linear_equation(B, w)

    # 4. Verification

    # a. Translation should be (0, 0, 2)
    # Why? It is the minimum norm solution to z=2.
    expected_trans = r3.from_components(np.array([0.0, 0.0, 2.0]))
    assert np.allclose(subspace.translation, expected_trans)

    # b. Projector should be onto XY plane (kernel of B)
    # Test vector (1, 1, 1) -> (1, 1, 0)
    v_test = r3.from_components(np.array([1.0, 1.0, 1.0]))
    v_tan = subspace.projector(v_test)
    expected_tan = r3.from_components(np.array([1.0, 1.0, 0.0]))
    assert np.allclose(v_tan, expected_tan)

    # c. Affine Projection
    # Project (5, 5, 10) onto plane z=2 -> (5, 5, 2)
    p_in = r3.from_components(np.array([5.0, 5.0, 10.0]))
    p_out = subspace.project(p_in)
    expected_out = r3.from_components(np.array([5.0, 5.0, 2.0]))
    assert np.allclose(p_out, expected_out)


def test_affine_constraint_function_space(circle_space):
    """
    Tests an affine subspace defined by a constraint C(f) = y.
    Constraint: Integral(f) = target_value.
    Uses the LinearSubspace.from_constraint (now from_kernel) and manual lifting.
    """
    # 1. Setup Constraint Operator C(f) = <f, 1>
    codomain = EuclideanSpace(1)
    ones_func = circle_space.project_function(lambda t: 1.0)

    def mapping(f):
        val = circle_space.inner_product(f, ones_func)
        return np.array([val])

    def adjoint_mapping(y):
        return circle_space.multiply(y[0], ones_func)

    C = LinearOperator(circle_space, codomain, mapping, adjoint_mapping=adjoint_mapping)

    # 2. Define Target Value
    norm_sq = circle_space.squared_norm(ones_func)
    target_mean = 5.0

    # Calculate a valid translation vector x0 manually
    translation = circle_space.multiply(target_mean / norm_sq, ones_func)

    # 3. Create Affine Subspace from Linear + Translation
    linear_subspace = LinearSubspace.from_kernel(C)
    affine_subspace = AffineSubspace(linear_subspace.projector, translation=translation)

    # 4. Verify translation is valid
    assert np.isclose(C(translation)[0], target_mean)
    assert affine_subspace.is_element(translation)

    # 5. Test Projection
    sine = circle_space.project_function(lambda t: np.sin(t))
    proj_sine = affine_subspace.project(sine)
    expected = circle_space.add(translation, sine)

    assert np.allclose(proj_sine, expected)
    # Verify the result satisfies the constraint C(u) = 5.0
    assert np.isclose(C(proj_sine)[0], target_mean)


def test_affine_subspace_bad_translation(r3):
    """
    Test that AffineSubspace raises an error if translation vector
    does not match the domain.
    """
    e1 = r3.basis_vector(0)
    subspace = LinearSubspace.from_basis(r3, [e1])

    # Create vector in R2
    r2 = EuclideanSpace(2)
    bad_vec = r2.from_components(np.array([1.0, 1.0]))

    # Attempt to init should raise ValueError because r3.is_element(bad_vec) is False
    with pytest.raises(ValueError, match="Translation vector"):
        AffineSubspace(subspace.projector, translation=bad_vec)


def test_affine_from_complement_basis_generates_constraints(r3):
    """
    Test that constructing an affine subspace from a complement basis
    automatically generates the corresponding constraint operator B and value w.

    Subspace: Plane z=2.
    Complement basis: e_z = [0, 0, 1].
    Translation: [0, 0, 2].

    Expected generated constraint:
    B(u) = <u, e_z>
    w = B(translation) = 2.
    """
    e_z = r3.basis_vector(2)
    translation = r3.from_components(np.array([0.0, 0.0, 2.0]))

    # 1. Construct using the geometric factory
    subspace = AffineSubspace.from_complement_basis(r3, [e_z], translation=translation)

    # 2. Check that the constraint equation was automatically generated
    assert subspace.has_constraint_equation

    # 3. Verify the operator B
    # It should map u -> <u, e_z> (coordinate projection)
    B = subspace.constraint_operator
    assert B.domain == r3
    assert B.codomain.dim == 1

    # Test action of B on a vector v = [1, 2, 3] -> should be 3
    v = r3.from_components(np.array([1.0, 2.0, 3.0]))
    b_v = B(v)
    assert np.isclose(b_v[0], 3.0)

    # 4. Verify the value w
    # Should be equal to B(translation) = 2
    w = subspace.constraint_value
    assert np.isclose(w[0], 2.0)

    # 5. Consistency Check
    # The subspace should be exactly the set {u | B(u) = w}
    # Test a point on the plane [5, 5, 2]
    p_in = r3.from_components(np.array([5.0, 5.0, 2.0]))
    assert np.allclose(B(p_in), w)


def test_linear_subspace_from_kernel_preserves_operator(r3):
    """
    Test that LinearSubspace.from_kernel preserves the operator B
    so it can be used for Bayesian updates.
    """
    # Operator B: u -> u_x + u_y + u_z
    codomain = EuclideanSpace(1)
    ones = np.array([[1.0, 1.0, 1.0]])
    B = LinearOperator.from_matrix(r3, codomain, ones)

    # Construct subspace
    subspace = LinearSubspace.from_kernel(B)

    # 1. Verify flag
    assert subspace.has_constraint_equation

    # 2. Verify stored operator is the one we passed
    assert subspace.constraint_operator is B

    # 3. Verify value is zero (since it's a LinearSubspace)
    w = subspace.constraint_value
    assert np.allclose(subspace.projector.codomain.to_components(w), 0.0)

    # 4. Verify geometry
    # (1, -1, 0) should be in kernel
    v_in = r3.from_components(np.array([1.0, -1.0, 0.0]))
    assert subspace.is_element(v_in)

    # (1, 1, 1) should not
    v_out = r3.from_components(np.array([1.0, 1.0, 1.0]))
    assert not subspace.is_element(v_out)
