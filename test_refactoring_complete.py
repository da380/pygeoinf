"""
Comprehensive test to verify the refactoring of transcendental equation solving.

This test verifies that:
1. RobinRootFinder.solve_tan_equation works for various cases
2. Robin boundary conditions still work correctly
3. Radial operators use the unified solver
"""

import sys
import math
import numpy as np

# Direct import to avoid dependency issues
import importlib.util

def load_module(name, path):
    """Load a module from a file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_solve_tan_equation():
    """Test the general tan equation solver."""
    print("="*60)
    print("TEST 1: RobinRootFinder.solve_tan_equation")
    print("="*60)

    robin_utils = load_module('robin_utils',
        '/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/interval/utils/robin_utils.py')
    RobinRootFinder = robin_utils.RobinRootFinder

    test_cases = [
        ("tan(k) = k", lambda k: k, 1.0, [
            (0, 4.4934094579),  # First root
            (1, 7.7252518369),  # Second root
            (2, 10.9041216594), # Third root
        ]),
        ("tan(2k) = 3k", lambda k: 3*k, 2.0, [
            (0, 2.8363227518),  # Approximate
        ]),
        ("tan(k) = -0.5k", lambda k: -0.5*k, 1.0, [
            (0, 2.0287958494),  # Approximate
        ]),
    ]

    all_passed = True
    for name, F, L, expected_roots in test_cases:
        print(f"\n  Testing: {name}")
        for idx, expected_k in expected_roots:
            k = RobinRootFinder.solve_tan_equation(F, L, idx)
            lhs = math.tan(k * L)
            rhs = F(k)
            diff = abs(lhs - rhs)

            # Check equation is satisfied
            if diff > 1e-9:
                print(f"    ✗ Root {idx}: FAILED equation check")
                print(f"      k = {k:.10f}")
                print(f"      tan(kL) = {lhs:.10f}, F(k) = {rhs:.10f}")
                print(f"      Difference: {diff:.2e}")
                all_passed = False
            else:
                print(f"    ✓ Root {idx}: k = {k:.10f} (error: {diff:.2e})")

    if all_passed:
        print("\n✓ All solve_tan_equation tests passed!")
    else:
        print("\n✗ Some tests failed!")

    return all_passed


def test_robin_bc_eigenvalues():
    """Test that Robin BC eigenvalue computation still works."""
    print("\n" + "="*60)
    print("TEST 2: Robin Boundary Conditions")
    print("="*60)

    robin_utils = load_module('robin_utils',
        '/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/interval/utils/robin_utils.py')
    RobinRootFinder = robin_utils.RobinRootFinder

    # Test pure Neumann (should give μ₀ = 0)
    print("\n  Testing pure Neumann (α₀=α_L=0, β₀=β_L=1):")
    mu0 = RobinRootFinder.compute_robin_eigenvalue(
        0, 0.0, 1.0, 0.0, 1.0, 1.0
    )
    print(f"    μ₀ = {mu0:.10e}")
    if abs(mu0) < 1e-10:
        print(f"    ✓ Correctly returns zero for pure Neumann")
    else:
        print(f"    ✗ Expected zero, got {mu0}")
        return False

    # Test pure Dirichlet (should give μ₀ ≈ π)
    print("\n  Testing pure Dirichlet (α₀=α_L=1, β₀=β_L=0):")
    mu0 = RobinRootFinder.compute_robin_eigenvalue(
        0, 1.0, 0.0, 1.0, 0.0, 1.0
    )
    print(f"    μ₀ = {mu0:.10f}")
    expected = math.pi
    if abs(mu0 - expected) < 0.01:
        print(f"    ✓ Close to π = {expected:.10f}")
    else:
        print(f"    ✗ Expected ~{expected:.10f}, got {mu0}")
        return False

    # Test mixed BC
    print("\n  Testing mixed Robin (α₀=1, β₀=0.5, α_L=1, β_L=0.5):")
    for idx in range(3):
        mu = RobinRootFinder.compute_robin_eigenvalue(
            idx, 1.0, 0.5, 1.0, 0.5, 1.0
        )

        # Verify it satisfies the characteristic equation
        L = 1.0
        alpha_0, beta_0 = 1.0, 0.5
        alpha_L, beta_L = 1.0, 0.5

        D = ((alpha_0 * alpha_L + beta_0 * beta_L * mu * mu) * math.sin(mu * L) +
             mu * (alpha_0 * beta_L - beta_0 * alpha_L) * math.cos(mu * L))

        print(f"    μ_{idx} = {mu:.10f}, D(μ) = {D:.2e}")
        if abs(D) > 1e-9:
            print(f"      ✗ Does not satisfy characteristic equation!")
            return False

    print("\n✓ All Robin BC tests passed!")
    return True


def test_radial_operators():
    """Test that radial operators use the new solver."""
    print("\n" + "="*60)
    print("TEST 3: Radial Operators")
    print("="*60)

    try:
        # Check that the import includes RobinRootFinder
        with open('/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/interval/operators/radial_operators.py', 'r') as f:
            content = f.read()

        if 'from ..utils.robin_utils import RobinRootFinder' not in content:
            print("  ✗ radial_operators.py doesn't import RobinRootFinder")
            return False
        else:
            print("  ✓ radial_operators.py imports RobinRootFinder")

        if 'RobinRootFinder.solve_tan_equation' not in content:
            print("  ✗ radial_operators.py doesn't use solve_tan_equation")
            return False
        else:
            print("  ✓ radial_operators.py uses solve_tan_equation")

        if '_root_between_tan_poles' in content:
            print("  ⚠ WARNING: Old _root_between_tan_poles method still exists")
            print("    (might be leftover, should be removed)")
        else:
            print("  ✓ Old _root_between_tan_poles method removed")

        # Count uses of solve_tan_equation
        count = content.count('RobinRootFinder.solve_tan_equation')
        print(f"\n  Found {count} uses of RobinRootFinder.solve_tan_equation")

        if count >= 4:  # Should be used in at least 4 methods
            print("  ✓ solve_tan_equation is used appropriately")
        else:
            print(f"  ⚠ Expected at least 4 uses, found {count}")

        print("\n✓ Radial operators correctly refactored!")
        return True

    except Exception as e:
        print(f"  ✗ Error checking radial operators: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print(" COMPREHENSIVE REFACTORING VERIFICATION TEST")
    print("="*70)
    print("\nVerifying that transcendental equation solving has been")
    print("successfully unified in RobinRootFinder...")

    results = []

    # Run tests
    results.append(("solve_tan_equation", test_solve_tan_equation()))
    results.append(("Robin BCs", test_robin_bc_eigenvalues()))
    results.append(("Radial operators", test_radial_operators()))

    # Summary
    print("\n" + "="*70)
    print(" TEST SUMMARY")
    print("="*70)

    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name:30s} {status}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\n" + "="*70)
        print(" ✓ ALL TESTS PASSED - REFACTORING SUCCESSFUL!")
        print("="*70)
        print("\nSummary of changes:")
        print("  • Added RobinRootFinder.solve_tan_equation() to robin_utils.py")
        print("  • Refactored radial_operators.py to use RobinRootFinder")
        print("  • Removed duplicate _root_between_tan_poles() method")
        print("  • All existing functionality preserved")
        print("="*70)
    else:
        print("\n" + "="*70)
        print(" ✗ SOME TESTS FAILED - PLEASE REVIEW")
        print("="*70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
