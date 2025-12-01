"""
Test the radial Laplacian eigenfunctions for (0, R) with regularity at 0.

Tests both Dirichlet and Neumann boundary conditions at R.
"""

import sys
import numpy as np
import math

# Direct import to avoid dependency issues
import importlib.util

def load_module(name, path):
    """Load a module from a file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def simpson_integrate(f, a, b, n=1000):
    """Simpson's rule integration."""
    if n % 2 == 1:
        n += 1
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    return h/3 * (y[0] + y[-1] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-1:2]))


def test_dirichlet_eigenfunctions():
    """Test regularity-Dirichlet eigenfunctions on (0, R)."""
    print("="*70)
    print("TEST 1: Regularity-Dirichlet eigenfunctions on (0, R)")
    print("="*70)

    # Load necessary modules
    sys.path.insert(0, '/home/adrian/PhD/Inferences/pygeoinf')
    from pygeoinf.interval import IntervalDomain, Lebesgue, BoundaryConditions
    from pygeoinf.interval.operators.radial_operators import (
        RadialLaplacianSpectrumProvider
    )

    R = 1.0
    interval = IntervalDomain(0.0, R)
    bc = BoundaryConditions('dirichlet')
    space = Lebesgue(5, interval)

    provider = RadialLaplacianSpectrumProvider(
        space, bc, alpha=1.0, inverse=False, ell=0
    )

    print(f"\nDomain: (0, {R})")
    print(f"Boundary condition: Dirichlet at r={R}")
    print("\nEigenvalues and eigenfunctions:")

    # Test first few eigenpairs
    all_passed = True
    for n in range(1, 4):  # n = 1, 2, 3
        index = n - 1  # 0-based indexing

        # Expected: k_n = nπ/R, λ_n = k_n²
        k_expected = n * math.pi / R
        lambda_expected = k_expected ** 2

        # Get from provider
        lambda_n = provider.get_eigenvalue(index)
        y_n = provider.get_eigenfunction(index)

        print(f"\n  Mode n={n} (index={index}):")
        print(f"    Expected: k_{n} = {k_expected:.10f}, λ_{n} = {lambda_expected:.10f}")
        print(f"    Computed: λ_{n} = {lambda_n:.10f}")
        print(f"    Error: {abs(lambda_n - lambda_expected):.2e}")

        if abs(lambda_n - lambda_expected) > 1e-10:
            print(f"    ✗ Eigenvalue mismatch!")
            all_passed = False
        else:
            print(f"    ✓ Eigenvalue correct")

        # Test orthonormality in L²((0,R); r²dr)
        # ∫₀ᴿ y_n² r² dr = 1
        r_test = np.linspace(0, R, 1000)
        y_vals = y_n(r_test)
        norm_sq = simpson_integrate(lambda r: y_n(r)**2 * r**2, 1e-10, R, n=1000)

        print(f"    Norm²: ∫ y_{n}² r² dr = {norm_sq:.10f}")
        print(f"    Error: {abs(norm_sq - 1.0):.2e}")

        if abs(norm_sq - 1.0) > 1e-3:
            print(f"    ✗ Normalization failed!")
            all_passed = False
        else:
            print(f"    ✓ Normalized correctly")

        # Test boundary condition: y(R) = φ(R)/R = 0 ⟹ φ(R) = 0
        # φ(r) = sqrt(2/R) sin(nπr/R), so φ(R) = sqrt(2/R) sin(nπ) = 0 ✓
        y_at_R = float(y_n(R))
        print(f"    BC check: y({R}) = {y_at_R:.2e}")

        # Since sin(nπ) = 0, y(R) should be 0/R which we evaluate as 0
        if abs(y_at_R) > 1e-6:
            print(f"    ⚠ Warning: y(R) should be ~0 for Dirichlet BC")

    if all_passed:
        print("\n✓ All Dirichlet tests passed!")
    else:
        print("\n✗ Some tests failed!")

    return all_passed


def test_neumann_eigenfunctions():
    """Test regularity-Neumann eigenfunctions on (0, R)."""
    print("\n" + "="*70)
    print("TEST 2: Regularity-Neumann eigenfunctions on (0, R)")
    print("="*70)

    # Load necessary modules
    sys.path.insert(0, '/home/adrian/PhD/Inferences/pygeoinf')
    from pygeoinf.interval import IntervalDomain, Lebesgue, BoundaryConditions
    from pygeoinf.interval.operators.radial_operators import (
        RadialLaplacianSpectrumProvider
    )

    R = 1.0
    interval = IntervalDomain(0.0, R)
    bc = BoundaryConditions('neumann')
    space = Lebesgue(5, interval)

    provider = RadialLaplacianSpectrumProvider(
        space, bc, alpha=1.0, inverse=False, ell=0
    )

    print(f"\nDomain: (0, {R})")
    print(f"Boundary condition: Neumann (flux) at r={R}")
    print("\nEigenvalues and eigenfunctions:")

    all_passed = True

    # Test zero mode (index=0)
    print(f"\n  Mode n=0 (zero mode):")
    lambda_0 = provider.get_eigenvalue(0)
    y_0 = provider.get_eigenfunction(0)

    print(f"    Expected: λ_0 = 0")
    print(f"    Computed: λ_0 = {lambda_0:.2e}")

    if abs(lambda_0) > 1e-10:
        print(f"    ✗ Zero eigenvalue mismatch!")
        all_passed = False
    else:
        print(f"    ✓ Zero eigenvalue correct")

    # y_0(r) should be constant c_0 = sqrt(3/R³)
    c_0_expected = np.sqrt(3.0 / R**3)
    r_test = np.linspace(0.1, R, 10)
    y_0_vals = y_0(r_test)

    print(f"    Expected: y_0(r) = {c_0_expected:.10f} (constant)")
    print(f"    Computed: y_0 values = {y_0_vals[:3]} ...")
    print(f"    Std dev: {np.std(y_0_vals):.2e}")

    if np.std(y_0_vals) > 1e-10:
        print(f"    ✗ Not constant!")
        all_passed = False
    else:
        print(f"    ✓ Constant as expected")

    # Check normalization: ∫₀ᴿ y_0² r² dr = 1
    norm_sq = simpson_integrate(lambda r: y_0(r)**2 * r**2, 1e-10, R, n=1000)
    print(f"    Norm²: ∫ y_0² r² dr = {norm_sq:.10f}")
    print(f"    Error: {abs(norm_sq - 1.0):.2e}")

    if abs(norm_sq - 1.0) > 1e-3:
        print(f"    ✗ Normalization failed!")
        all_passed = False
    else:
        print(f"    ✓ Normalized correctly")

    # Test nonzero modes
    for n in range(1, 4):  # n = 1, 2, 3
        index = n

        print(f"\n  Mode n={n} (index={index}):")

        lambda_n = provider.get_eigenvalue(index)
        k_n = np.sqrt(lambda_n)
        y_n = provider.get_eigenfunction(index)

        # Expected: k_n R ≈ (n + 1/2)π asymptotically
        k_expected_approx = (n + 0.5) * math.pi / R

        print(f"    k_{n} = {k_n:.10f}")
        print(f"    Approx: k_{n} ≈ {k_expected_approx:.10f}")
        print(f"    λ_{n} = {lambda_n:.10f}")

        # Check that tan(k_n R) = k_n R
        zeta_n = k_n * R
        lhs = math.tan(zeta_n)
        rhs = zeta_n
        print(f"    Transcendental: tan(k_n R) = {lhs:.10f}, k_n R = {rhs:.10f}")
        print(f"    Error: {abs(lhs - rhs):.2e}")

        if abs(lhs - rhs) > 1e-8:
            print(f"    ✗ Transcendental equation not satisfied!")
            all_passed = False
        else:
            print(f"    ✓ Transcendental equation satisfied")

        # Test orthonormality in L²((0,R); r²dr)
        norm_sq = simpson_integrate(lambda r: y_n(r)**2 * r**2, 1e-10, R, n=1000)
        print(f"    Norm²: ∫ y_{n}² r² dr = {norm_sq:.10f}")
        print(f"    Error: {abs(norm_sq - 1.0):.2e}")

        if abs(norm_sq - 1.0) > 1e-3:
            print(f"    ✗ Normalization failed!")
            all_passed = False
        else:
            print(f"    ✓ Normalized correctly")

        # Test boundary condition: u'(R) = u(R)/R
        # u(r) = φ_n(r) = c_n sin(k_n r)
        # u'(r) = c_n k_n cos(k_n r)
        # At r=R: c_n k_n cos(k_n R) = c_n sin(k_n R)/R
        # ⟹ k_n R cos(k_n R) = sin(k_n R)
        # ⟹ tan(k_n R) = k_n R (already checked above)

    # Test orthogonality between modes
    print(f"\n  Testing orthogonality:")
    y_1 = provider.get_eigenfunction(1)
    y_2 = provider.get_eigenfunction(2)

    inner_prod = simpson_integrate(
        lambda r: y_1(r) * y_2(r) * r**2, 1e-10, R, n=1000
    )
    print(f"    ⟨y_1, y_2⟩ = {inner_prod:.2e}")

    if abs(inner_prod) > 1e-3:
        print(f"    ✗ Orthogonality failed!")
        all_passed = False
    else:
        print(f"    ✓ Orthogonal")

    if all_passed:
        print("\n✓ All Neumann tests passed!")
    else:
        print("\n✗ Some tests failed!")

    return all_passed


def test_dirichlet_dirichlet_ab():
    """Test Dirichlet-Dirichlet eigenfunctions on (a, b) with a > 0."""
    print("\n" + "="*70)
    print("TEST 3: Dirichlet-Dirichlet eigenfunctions on (a, b)")
    print("="*70)

    # Load necessary modules
    sys.path.insert(0, '/home/adrian/PhD/Inferences/pygeoinf')
    from pygeoinf.interval import IntervalDomain, Lebesgue, BoundaryConditions
    from pygeoinf.interval.operators.radial_operators import (
        RadialLaplacianSpectrumProvider
    )

    a, b = 0.5, 2.0
    L = b - a
    interval = IntervalDomain(a, b)
    bc = BoundaryConditions('dirichlet')
    space = Lebesgue(5, interval)

    provider = RadialLaplacianSpectrumProvider(
        space, bc, alpha=1.0, inverse=False, ell=0
    )

    print(f"\nDomain: ({a}, {b}), L = {L}")
    print(f"Boundary condition: Dirichlet at both ends")
    print("\nEigenvalues and eigenfunctions:")

    all_passed = True

    # Test first few eigenpairs
    for n in range(1, 4):  # n = 1, 2, 3
        index = n - 1  # 0-based indexing

        # Expected: k_n = nπ/L, λ_n = k_n²
        k_expected = n * math.pi / L
        lambda_expected = k_expected ** 2

        # Get from provider
        lambda_n = provider.get_eigenvalue(index)
        y_n = provider.get_eigenfunction(index)

        print(f"\n  Mode n={n} (index={index}):")
        print(f"    Expected: k_{n} = {k_expected:.10f}, λ_{n} = {lambda_expected:.10f}")
        print(f"    Computed: λ_{n} = {lambda_n:.10f}")
        print(f"    Error: {abs(lambda_n - lambda_expected):.2e}")

        if abs(lambda_n - lambda_expected) > 1e-10:
            print(f"    ✗ Eigenvalue mismatch!")
            all_passed = False
        else:
            print(f"    ✓ Eigenvalue correct")

        # Test orthonormality in L²((a,b); r²dr)
        # ∫ₐᵇ y_n² r² dr = 1
        norm_sq = simpson_integrate(lambda r: y_n(r)**2 * r**2, a, b, n=1000)

        print(f"    Norm²: ∫ y_{n}² r² dr = {norm_sq:.10f}")
        print(f"    Error: {abs(norm_sq - 1.0):.2e}")

        if abs(norm_sq - 1.0) > 1e-3:
            print(f"    ✗ Normalization failed!")
            all_passed = False
        else:
            print(f"    ✓ Normalized correctly")

        # Test boundary conditions: y(a) ≈ 0 and y(b) ≈ 0
        # φ(r) = sqrt(2/L) sin(nπ(r-a)/L)
        # At r=a: φ(a) = 0 ✓
        # At r=b: φ(b) = sqrt(2/L) sin(nπ) = 0 ✓
        y_at_a = float(y_n(a))
        y_at_b = float(y_n(b))
        print(f"    BC check: y({a}) = {y_at_a:.2e}, y({b}) = {y_at_b:.2e}")

        # Both should be close to 0
        if abs(y_at_a) > 1e-6 or abs(y_at_b) > 1e-6:
            print(f"    ⚠ Warning: Boundary values should be ~0 for Dirichlet BC")

    # Test orthogonality between modes
    print(f"\n  Testing orthogonality:")
    y_1 = provider.get_eigenfunction(0)
    y_2 = provider.get_eigenfunction(1)

    inner_prod = simpson_integrate(
        lambda r: y_1(r) * y_2(r) * r**2, a, b, n=1000
    )
    print(f"    ⟨y_1, y_2⟩ = {inner_prod:.2e}")

    if abs(inner_prod) > 1e-3:
        print(f"    ✗ Orthogonality failed!")
        all_passed = False
    else:
        print(f"    ✓ Orthogonal")

    if all_passed:
        print("\n✓ All Dirichlet-Dirichlet tests passed!")
    else:
        print("\n✗ Some tests failed!")

    return all_passed


def test_mixed_bc_cases():
    """Test DN, ND, and NN cases on (a, b) with a > 0."""
    print("\n" + "="*70)
    print("TEST 4: Mixed Boundary Conditions on (a, b)")
    print("="*70)

    sys.path.insert(0, '/home/adrian/PhD/Inferences/pygeoinf')
    from pygeoinf.interval import IntervalDomain, Lebesgue, BoundaryConditions
    from pygeoinf.interval.operators.radial_operators import (
        RadialLaplacianSpectrumProvider
    )

    a, b = 0.5, 2.0
    L = b - a
    all_passed = True

    # Test DN (Dirichlet-Neumann)
    print(f"\n  === Testing DN (Dirichlet at a={a}, Neumann at b={b}) ===")
    interval_dn = IntervalDomain(a, b)
    bc_dn = BoundaryConditions('mixed_dirichlet_neumann')
    space_dn = Lebesgue(5, interval_dn)
    provider_dn = RadialLaplacianSpectrumProvider(
        space_dn, bc_dn, alpha=1.0, inverse=False, ell=0
    )

    for n in range(1, 3):
        index = n - 1
        lambda_n = provider_dn.get_eigenvalue(index)
        k_n = np.sqrt(lambda_n)
        y_n = provider_dn.get_eigenfunction(index)

        # Check transcendental equation: tan(kL) = kb
        lhs = math.tan(k_n * L)
        rhs = k_n * b
        print(f"    Mode {n}: tan(k_n L) = {lhs:.6f}, k_n b = {rhs:.6f}, " +
              f"error = {abs(lhs - rhs):.2e}")

        # Check normalization
        norm_sq = simpson_integrate(lambda r: y_n(r)**2 * r**2, a, b, n=1000)
        print(f"             Norm² = {norm_sq:.10f}, error = {abs(norm_sq - 1.0):.2e}")

        if abs(lhs - rhs) > 1e-8 or abs(norm_sq - 1.0) > 1e-3:
            all_passed = False

    # Test ND (Neumann-Dirichlet)
    print(f"\n  === Testing ND (Neumann at a={a}, Dirichlet at b={b}) ===")
    interval_nd = IntervalDomain(a, b)
    bc_nd = BoundaryConditions('mixed_neumann_dirichlet')
    space_nd = Lebesgue(5, interval_nd)
    provider_nd = RadialLaplacianSpectrumProvider(
        space_nd, bc_nd, alpha=1.0, inverse=False, ell=0
    )

    for n in range(1, 3):
        index = n - 1
        lambda_n = provider_nd.get_eigenvalue(index)
        k_n = np.sqrt(lambda_n)
        y_n = provider_nd.get_eigenfunction(index)

        # Check transcendental equation: tan(kL) = -ak
        lhs = math.tan(k_n * L)
        rhs = -a * k_n
        print(f"    Mode {n}: tan(k_n L) = {lhs:.6f}, -a k_n = {rhs:.6f}, " +
              f"error = {abs(lhs - rhs):.2e}")

        # Check normalization
        norm_sq = simpson_integrate(lambda r: y_n(r)**2 * r**2, a, b, n=1000)
        print(f"             Norm² = {norm_sq:.10f}, error = {abs(norm_sq - 1.0):.2e}")

        if abs(lhs - rhs) > 1e-8 or abs(norm_sq - 1.0) > 1e-3:
            all_passed = False

    # Test NN (Neumann-Neumann)
    print(f"\n  === Testing NN (Neumann at both a={a} and b={b}) ===")
    interval_nn = IntervalDomain(a, b)
    bc_nn = BoundaryConditions('neumann')
    space_nn = Lebesgue(5, interval_nn)
    provider_nn = RadialLaplacianSpectrumProvider(
        space_nn, bc_nn, alpha=1.0, inverse=False, ell=0
    )

    # Test zero mode
    lambda_0 = provider_nn.get_eigenvalue(0)
    y_0 = provider_nn.get_eigenfunction(0)
    print(f"    Zero mode: λ_0 = {lambda_0:.2e}")

    # Check if constant
    r_test = np.linspace(a+0.1, b-0.1, 10)
    y_0_vals = y_0(r_test)
    print(f"               y_0 values std = {np.std(y_0_vals):.2e} (should be ~0)")

    # Check normalization
    norm_sq = simpson_integrate(lambda r: y_0(r)**2 * r**2, a, b, n=1000)
    print(f"               Norm² = {norm_sq:.10f}, error = {abs(norm_sq - 1.0):.2e}")

    if abs(lambda_0) > 1e-10 or np.std(y_0_vals) > 1e-10 or abs(norm_sq - 1.0) > 1e-3:
        all_passed = False

    # Test nonzero modes
    for n in range(1, 3):
        index = n
        lambda_n = provider_nn.get_eigenvalue(index)
        k_n = np.sqrt(lambda_n)
        y_n = provider_nn.get_eigenfunction(index)

        # Check transcendental equation: tan(kL) = (1/b - 1/a)/(k + 1/(abk))
        lhs = math.tan(k_n * L)
        rhs = (1.0/b - 1.0/a) / (k_n + 1.0/(a*b*k_n))
        print(f"    Mode {n}: tan(k_n L) = {lhs:.6f}, RHS = {rhs:.6f}, " +
              f"error = {abs(lhs - rhs):.2e}")

        # Check normalization
        norm_sq = simpson_integrate(lambda r: y_n(r)**2 * r**2, a, b, n=1000)
        print(f"             Norm² = {norm_sq:.10f}, error = {abs(norm_sq - 1.0):.2e}")

        if abs(lhs - rhs) > 1e-8 or abs(norm_sq - 1.0) > 1e-3:
            all_passed = False

    if all_passed:
        print("\n✓ All mixed BC tests passed!")
    else:
        print("\n✗ Some tests failed!")

    return all_passed


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print(" RADIAL LAPLACIAN EIGENFUNCTION TESTS")
    print("="*70)
    print("\nTesting eigenfunctions y_n(r) orthonormal in L²((a,b); r²dr)")

    results = []

    try:
        results.append(("(0,R) Dirichlet", test_dirichlet_eigenfunctions()))
    except Exception as e:
        print(f"\n✗ Dirichlet test failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("(0,R) Dirichlet", False))

    try:
        results.append(("(0,R) Neumann", test_neumann_eigenfunctions()))
    except Exception as e:
        print(f"\n✗ Neumann test failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("(0,R) Neumann", False))

    try:
        results.append(("(a,b) Dirichlet-Dirichlet", test_dirichlet_dirichlet_ab()))
    except Exception as e:
        print(f"\n✗ Dirichlet-Dirichlet test failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("(a,b) Dirichlet-Dirichlet", False))

    try:
        results.append(("(a,b) DN/ND/NN", test_mixed_bc_cases()))
    except Exception as e:
        print(f"\n✗ Mixed BC test failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("(a,b) DN/ND/NN", False))

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
        print(" ✓ ALL TESTS PASSED!")
        print("="*70)
    else:
        print("\n" + "="*70)
        print(" ✗ SOME TESTS FAILED")
        print("="*70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
