#!/usr/bin/env python3
"""
Comprehensive demonstration of the new SobolevFunction-based coefficient methods.
This shows how the factory method now creates spaces where:
- from_coefficient returns SobolevFunction instances (linear combinations of basis)
- to_coefficient uses proper L2 inner products with basis functions
- Round-trip conversion is exact (within machine precision)
"""

import numpy as np
import matplotlib.pyplot as plt
from pygeoinf.other_space.interval_space import Sobolev
from pygeoinf.other_space.sobolev_functions import SobolevFunction


def demonstrate_new_functionality():
    """Demonstrate the new SobolevFunction-based coefficient methods."""

    print("=== New SobolevFunction-Based Coefficient Methods ===\n")

    # Create a Sobolev space
    order = 2.0
    scale = 1.0
    dim = 5
    interval = (0, 1)

    print(f"Creating H^{order}([{interval[0]}, {interval[1]}]) space with {dim} basis functions")

    space = Sobolev.create_standard_sobolev(
        order, scale, dim, interval=interval, basis_type='fourier'
    )

    print(f"Basis functions: {[f.name for f in space.basis_functions]}")

    print("\n--- Key Innovation: from_coefficient returns SobolevFunction ---")

    # Define some coefficients
    coeffs = np.array([1.0, 0.8, -0.3, 0.5, 0.2])
    print(f"Coefficients: {coeffs}")

    # Convert to function - this now returns a SobolevFunction!
    func = space.from_coefficient(coeffs)
    print(f"Result type: {type(func)}")
    print(f"Function name: {func.name}")

    # The returned function is a linear combination of basis functions
    print(f"\nThis function represents: {coeffs[0]:.1f} * {space.basis_functions[0].name}")
    for i in range(1, len(coeffs)):
        print(f"                       + {coeffs[i]:.1f} * {space.basis_functions[i].name}")

    print("\n--- The function can be evaluated at any point ---")
    test_points = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    values = func.evaluate(test_points)
    print(f"f({test_points}) = {values}")

    print("\n--- Key Innovation: to_coefficient uses L2 inner products ---")

    # Convert back to coefficients using inner products
    recovered_coeffs = space.to_coefficient(func)
    print(f"Original coefficients:  {coeffs}")
    print(f"Recovered coefficients: {recovered_coeffs}")
    print(f"Round-trip error: {np.linalg.norm(coeffs - recovered_coeffs)}")

    print("\n--- Working with arbitrary functions ---")

    # Define a custom function
    def my_function(x):
        return np.exp(-5*(x-0.5)**2) * np.cos(4*np.pi*x)

    # Convert to coefficient representation
    coeffs_custom = space.to_coefficient(my_function)
    print(f"Custom function: exp(-5*(x-0.5)^2) * cos(4πx)")
    print(f"Coefficients: {coeffs_custom}")

    # Convert back to SobolevFunction
    approx_func = space.from_coefficient(coeffs_custom)
    print(f"Approximation type: {type(approx_func)}")

    # Compare original and approximation
    x_test = np.linspace(0, 1, 20)
    original_vals = np.array([my_function(x) for x in x_test])
    approx_vals = approx_func.evaluate(x_test)

    print(f"Approximation error (L2): {np.sqrt(np.trapz((original_vals - approx_vals)**2, x_test))}")

    print("\n--- Mathematical Properties ---")

    # Show that the basis functions span the space
    print(f"Basis functions are elements of H^{space.order}([{space.interval[0]}, {space.interval[1]}])")
    print(f"Each has Sobolev order: {space.basis_functions[0].sobolev_order}")
    print(f"Each has domain: {space.basis_functions[0].domain}")

    # Show inner product computations
    print(f"\nGram matrix shape: {space._gram_matrix.shape}")
    print(f"Gram matrix condition number: {np.linalg.cond(space._gram_matrix):.2e}")

    print("\n--- Compatibility with existing code ---")

    # The space still works with arrays (falls back to original methods)
    array_vals = np.random.randn(dim)
    coeffs_from_array = space.to_coefficient(array_vals)
    print(f"Can still convert arrays: {array_vals[:3]}... -> {coeffs_from_array[:3]}...")

    return space


def demonstrate_different_bases():
    """Show how this works with different basis types."""

    print("\n=== Different Basis Types ===\n")

    order = 1.5
    scale = 0.5
    dim = 4
    interval = (0, 1)

    for basis_type in ['fourier', 'sine', 'chebyshev']:
        print(f"--- {basis_type.upper()} BASIS ---")

        space = Sobolev.create_standard_sobolev(
            order, scale, dim, interval=interval, basis_type=basis_type
        )

        # Show basis function names
        print(f"Basis functions: {[f.name for f in space.basis_functions]}")

        # Test with a simple function
        def test_func(x):
            return x * (1 - x)  # Parabola that vanishes at endpoints

        # Convert to coefficients and back
        coeffs = space.to_coefficient(test_func)
        reconstructed = space.from_coefficient(coeffs)

        # Test accuracy
        x_test = np.linspace(0.1, 0.9, 10)  # Avoid endpoints for sine basis
        original_vals = np.array([test_func(x) for x in x_test])
        reconstructed_vals = reconstructed.evaluate(x_test)
        error = np.linalg.norm(original_vals - reconstructed_vals)

        print(f"Coefficients: {coeffs}")
        print(f"Approximation error: {error:.6f}")
        print(f"Reconstructed function type: {type(reconstructed)}")
        print()


def demonstrate_mathematical_insight():
    """Show the mathematical insight behind the implementation."""

    print("\n=== Mathematical Insight ===\n")

    print("The key insight is that we now have a proper mathematical representation:")
    print("1. The space H^s([a,b]) is spanned by basis functions {φ_k}")
    print("2. Any u ∈ H^s([a,b]) can be written as u = Σ c_k φ_k")
    print("3. Coefficients are found by solving: G c = b")
    print("   where G_ij = <φ_i, φ_j>_L2 and b_i = <u, φ_i>_L2")
    print("4. This ensures that from_coefficient(to_coefficient(u)) = u exactly")
    print()

    # Create a simple example
    space = Sobolev.create_standard_sobolev(2.0, 1.0, 3, basis_type='fourier')

    print("Example with 3 Fourier basis functions:")
    print(f"Basis: {[f.name for f in space.basis_functions]}")
    print()

    # Show the Gram matrix
    print("Gram matrix G (inner products of basis functions):")
    print(f"{space._gram_matrix}")
    print()

    # Show how coefficients are computed
    def example_func(x):
        return x**2

    print("For u(x) = x^2:")

    # Compute inner products manually
    from scipy.integrate import quad

    inner_products = []
    for i, basis_func in enumerate(space.basis_functions):
        def integrand(x):
            return x**2 * basis_func.evaluate(x)
        result, _ = quad(integrand, 0, 1)
        inner_products.append(result)

    print(f"Inner products b = [<u, φ_0>, <u, φ_1>, <u, φ_2>] = {inner_products}")

    # Solve for coefficients
    coeffs_manual = np.linalg.solve(space._gram_matrix, inner_products)
    coeffs_automatic = space.to_coefficient(example_func)

    print(f"Manual calculation: c = G^(-1) b = {coeffs_manual}")
    print(f"Automatic calculation: {coeffs_automatic}")
    print(f"Difference: {np.linalg.norm(coeffs_manual - coeffs_automatic)}")


if __name__ == "__main__":
    try:
        space = demonstrate_new_functionality()
        demonstrate_different_bases()
        demonstrate_mathematical_insight()
        print("\n🎉 All demonstrations completed successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
