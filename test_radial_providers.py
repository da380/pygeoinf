"""
Test radial Laplacian function providers.

This tests that:
1. Function providers can be used independently
2. Function providers can be used as basis in Lebesgue spaces
3. Results match the RadialLaplacian operator eigenfunctions
"""

import numpy as np
from pygeoinf.interval import IntervalDomain, Lebesgue, BoundaryConditions
from pygeoinf.interval.function_providers import (
    RadialLaplacianDirichletProvider,
    RadialLaplacianNeumannProvider,
    RadialLaplacianDDProvider,
    RadialLaplacianDNProvider,
    RadialLaplacianNDProvider,
    RadialLaplacianNNProvider,
)
from pygeoinf.interval.operators import RadialLaplacian
from pygeoinf.interval.configs import IntegrationConfig


def simpson_integrate(f, a, b, n=1000):
    """Simpson's rule integration."""
    x = np.linspace(a, b, n)
    y = f(x)
    h = (b - a) / (n - 1)
    return h / 3 * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]) + y[-1])


def test_provider_independently():
    """Test that providers work independently without operators."""
    print("\n" + "="*70)
    print("TEST 1: Function providers work independently")
    print("="*70)

    # Domain (0, 2) with Dirichlet BC
    domain = IntervalDomain(0.0, 2.0)
    bc = BoundaryConditions(bc_type='dirichlet')
    space = Lebesgue(50, domain)

    # Create provider
    provider = RadialLaplacianDirichletProvider(space)

    # Get first few eigenfunctions
    for i in range(3):
        func = provider.get_function_by_index(i)
        print(f"  Mode {i}: {func.name}")

        # Test orthonormality
        for j in range(i+1):
            func_j = provider.get_function_by_index(j)
            inner = simpson_integrate(
                lambda r: func.evaluate(r) * func_j.evaluate(r) * r**2,
                domain.a, domain.b
            )
            expected = 1.0 if i == j else 0.0
            error = abs(inner - expected)
            print(f"    ⟨y_{i}, y_{j}⟩ = {inner:.10f}, error = {error:.2e}")
            assert error < 1e-3, f"Orthonormality failed: error {error}"

    print("✓ Provider works independently!")


def test_provider_with_lebesgue_space():
    """Test that providers can be used as basis in Lebesgue spaces."""
    print("\n" + "="*70)
    print("TEST 2: Function providers as basis in Lebesgue space")
    print("="*70)

    # Test different basis types
    test_cases = [
        ('radial_dirichlet', IntervalDomain(0.0, 2.0), 'dirichlet'),
        ('radial_neumann', IntervalDomain(0.0, 2.0), 'neumann'),
        ('radial_DD', IntervalDomain(0.5, 2.0), 'dirichlet'),
        ('radial_DN', IntervalDomain(0.5, 2.0), 'mixed_dirichlet_neumann'),
        ('radial_ND', IntervalDomain(0.5, 2.0), 'mixed_neumann_dirichlet'),
        ('radial_NN', IntervalDomain(0.5, 2.0), 'neumann'),
    ]

    for basis_type, domain, bc_type in test_cases:
        print(f"\n  Testing basis_type='{basis_type}' on domain ({domain.a}, {domain.b})")

        # Create space with radial basis
        space = Lebesgue(30, domain, basis=basis_type)

        # Get first few basis functions
        for i in range(min(3, space.dim)):
            func = space.get_basis_function(i)

            # Test that it's a valid function
            r_test = np.linspace(domain.a + 0.01, domain.b - 0.01, 10)
            values = func.evaluate(r_test)
            assert np.all(np.isfinite(values)), f"Non-finite values in basis function {i}"

        print(f"    ✓ All basis functions valid")

    print("\n✓ All basis types work in Lebesgue space!")


def test_consistency_with_operator():
    """Test that provider functions match operator eigenfunctions."""
    print("\n" + "="*70)
    print("TEST 3: Consistency with RadialLaplacian operator")
    print("="*70)

    # Test (0, R) Dirichlet case
    print("\n  Testing (0, R) Dirichlet case:")
    domain = IntervalDomain(0.0, 2.0)
    bc = BoundaryConditions(bc_type='dirichlet')
    int_cfg = IntegrationConfig(method='gauss', n_points=500)
    space = Lebesgue(50, domain, integration_config=int_cfg)

    # Create operator and provider
    operator = RadialLaplacian(
        space, bc, method='spectral',
        integration_config=int_cfg
    )
    provider = RadialLaplacianDirichletProvider(space)

    # Compare eigenfunctions
    r_test = np.linspace(domain.a + 0.01, domain.b, 100)
    for i in range(3):
        func_op = operator.get_eigenfunction(i)
        func_prov = provider.get_function_by_index(i)

        values_op = func_op.evaluate(r_test)
        values_prov = func_prov.evaluate(r_test)

        # They should be identical (or negatives of each other)
        diff = np.abs(values_op - values_prov)
        diff_neg = np.abs(values_op + values_prov)
        error = min(np.max(diff), np.max(diff_neg))

        print(f"    Mode {i}: max difference = {error:.2e}")
        assert error < 1e-10, f"Functions don't match: error {error}"

    print("    ✓ Functions match operator eigenfunctions")

    # Test (a, b) NN case
    print("\n  Testing (a, b) Neumann-Neumann case:")
    domain = IntervalDomain(0.5, 2.0)
    bc = BoundaryConditions(bc_type='neumann')
    space = Lebesgue(50, domain, integration_config=int_cfg)

    operator = RadialLaplacian(
        space, bc, method='spectral',
        integration_config=int_cfg
    )
    provider = RadialLaplacianNNProvider(space)

    r_test = np.linspace(domain.a, domain.b, 100)
    for i in range(3):
        func_op = operator.get_eigenfunction(i)
        func_prov = provider.get_function_by_index(i)

        values_op = func_op.evaluate(r_test)
        values_prov = func_prov.evaluate(r_test)

        diff = np.abs(values_op - values_prov)
        diff_neg = np.abs(values_op + values_prov)
        error = min(np.max(diff), np.max(diff_neg))

        print(f"    Mode {i}: max difference = {error:.2e}")
        assert error < 1e-10, f"Functions don't match: error {error}"

    print("    ✓ Functions match operator eigenfunctions")

    print("\n✓ Provider functions consistent with operator!")


if __name__ == "__main__":
    test_provider_independently()
    test_provider_with_lebesgue_space()
    test_consistency_with_operator()

    print("\n" + "="*70)
    print(" ALL TESTS PASSED!")
    print("="*70)
