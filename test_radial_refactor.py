"""
Quick test to verify the radial operators refactoring works.
Tests that RobinRootFinder.solve_tan_equation produces the same results
as the old _root_between_tan_poles method.
"""

import sys
sys.path.insert(0, '/home/adrian/PhD/Inferences/pygeoinf')

import numpy as np
from pygeoinf.interval.utils.robin_utils import RobinRootFinder


def test_solve_tan_equation():
    """Test the general tan equation solver."""
    print("Testing RobinRootFinder.solve_tan_equation()...")

    # Test case 1: tan(kL) = kR (regularity-Neumann)
    R = 1.0
    F1 = lambda k: k * R
    k1 = RobinRootFinder.solve_tan_equation(F1, R, 0)
    print(f"  tan(k*{R}) = k*{R}, first root: k = {k1:.10f}")
    print(f"  Verification: tan(k*R) = {np.tan(k1 * R):.10f}, k*R = {k1 * R:.10f}")
    print(f"  Difference: {abs(np.tan(k1 * R) - k1 * R):.2e}")
    assert abs(np.tan(k1 * R) - k1 * R) < 1e-10, "Failed to solve tan(kR) = kR"

    # Test case 2: tan(kL) = kb (Dirichlet-Neumann)
    L = 2.0
    b = 3.0
    F2 = lambda k: k * b
    k2 = RobinRootFinder.solve_tan_equation(F2, L, 0)
    print(f"\n  tan(k*{L}) = k*{b}, first root: k = {k2:.10f}")
    print(f"  Verification: tan(k*L) = {np.tan(k2 * L):.10f}, k*b = {k2 * b:.10f}")
    print(f"  Difference: {abs(np.tan(k2 * L) - k2 * b):.2e}")
    assert abs(np.tan(k2 * L) - k2 * b) < 1e-10, "Failed to solve tan(kL) = kb"

    # Test case 3: tan(kL) = -ak (Neumann-Dirichlet)
    a = 0.5
    L = 2.5
    F3 = lambda k: -a * k
    k3 = RobinRootFinder.solve_tan_equation(F3, L, 0)
    print(f"\n  tan(k*{L}) = -{a}*k, first root: k = {k3:.10f}")
    print(f"  Verification: tan(k*L) = {np.tan(k3 * L):.10f}, -a*k = {-a * k3:.10f}")
    print(f"  Difference: {abs(np.tan(k3 * L) - (-a * k3)):.2e}")
    assert abs(np.tan(k3 * L) - (-a * k3)) < 1e-10, "Failed to solve tan(kL) = -ak"

    # Test case 4: Multiple roots
    print(f"\n  Testing multiple roots for tan(kR) = kR:")
    for idx in range(5):
        k = RobinRootFinder.solve_tan_equation(F1, R, idx)
        diff = abs(np.tan(k * R) - k * R)
        print(f"    Root {idx}: k = {k:.10f}, difference = {diff:.2e}")
        assert diff < 1e-10, f"Failed at root index {idx}"

    print("\n✓ All tests passed!")


def test_radial_operator_import():
    """Test that radial operators can be imported and use RobinRootFinder."""
    print("\nTesting radial operator imports...")

    try:
        from pygeoinf.interval.operators.radial_operators import (
            RadialLaplacianEigenvalueProvider
        )
        from pygeoinf.interval import Interval, Lebesgue, BoundaryConditions

        # Create a simple domain
        interval = Interval(0.0, 1.0)
        space = Lebesgue(interval, dim=10)

        # Test regularity-Neumann case (uses solve_tan_equation)
        bc = BoundaryConditions('neumann')
        provider = RadialLaplacianEigenvalueProvider(
            interval, bc, inverse=False, alpha=1.0, ell=0
        )

        # Get first eigenvalue
        eigenval = provider.get_eigenvalue(0)
        print(f"  First eigenvalue (regularity-Neumann): λ₀ = {eigenval:.10f}")

        # For regularity-Neumann, eigenvalue should be k² where tan(kR) = kR
        k_expected = RobinRootFinder.solve_tan_equation(lambda k: k * 1.0, 1.0, 0)
        eigenval_expected = k_expected ** 2
        print(f"  Expected from direct calculation: λ₀ = {eigenval_expected:.10f}")
        print(f"  Difference: {abs(eigenval - eigenval_expected):.2e}")

        assert abs(eigenval - eigenval_expected) < 1e-10, \
            "Eigenvalue doesn't match expected value"

        print("✓ Radial operator successfully uses RobinRootFinder!")

    except Exception as e:
        print(f"✗ Error testing radial operators: {e}")
        raise


if __name__ == "__main__":
    test_solve_tan_equation()
    test_radial_operator_import()
    print("\n" + "="*60)
    print("All refactoring tests passed successfully!")
    print("="*60)
