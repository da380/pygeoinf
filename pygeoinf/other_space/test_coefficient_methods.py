#!/usr/bin/env python3
"""
Test the new coefficient methods that use basis functions and inner products.
"""

import numpy as np
from pygeoinf.other_space.interval_space import Sobolev
from pygeoinf.other_space.sobolev_functions import SobolevFunction


def test_basis_coefficient_methods():
    """Test the new coefficient methods using basis functions."""

    print("=== Testing Basis-Function Coefficient Methods ===\n")

    # Create a simple space
    order = 2.0  # High enough for point evaluation
    scale = 1.0
    dim = 4
    interval = (0, 1)

    print(f"Creating Sobolev space H^{order}([{interval[0]}, {interval[1]}])")
    print(f"Dimension: {dim}")

    # Test with Fourier basis
    space = Sobolev.create_standard_sobolev(
        order, scale, dim, interval=interval, basis_type='fourier'
    )

    print(f"Created space with {len(space.basis_functions)} basis functions")

    # Test 1: from_coefficient should return a SobolevFunction
    print("\n--- Test 1: from_coefficient returns SobolevFunction ---")
    test_coeffs = np.array([1.0, 0.5, 0.0, 0.2])

    result_func = space.from_coefficient(test_coeffs)
    print(f"Coefficients: {test_coeffs}")
    print(f"Result type: {type(result_func)}")
    print(f"Is SobolevFunction: {isinstance(result_func, SobolevFunction)}")

    if isinstance(result_func, SobolevFunction):
        # Test evaluation
        test_points = [0.0, 0.25, 0.5, 0.75, 1.0]
        values = [result_func.evaluate(x) for x in test_points]
        print(f"Values at {test_points}: {values}")

        # Manual verification: should equal linear combination
        manual_values = []
        for x in test_points:
            manual_val = 0.0
            for k, c in enumerate(test_coeffs):
                manual_val += c * space.basis_functions[k].evaluate(x)
            manual_values.append(manual_val)

        print(f"Manual calculation: {manual_values}")
        print(f"Difference: {[abs(a-b) for a, b in zip(values, manual_values)]}")

    # Test 2: to_coefficient with a SobolevFunction
    print("\n--- Test 2: to_coefficient with SobolevFunction ---")

    # Create a test function (simple polynomial)
    def test_function(x):
        return x**2 - 0.5

    test_sobolev_func = SobolevFunction(
        space, evaluate_callable=test_function, name="x^2 - 0.5"
    )

    coeffs = space.to_coefficient(test_sobolev_func)
    print(f"Function: {test_sobolev_func.name}")
    print(f"Computed coefficients: {coeffs}")

    # Test 3: Round-trip test
    print("\n--- Test 3: Round-trip test ---")

    # Start with coefficients, convert to function, then back to coefficients
    original_coeffs = np.array([0.8, -0.3, 0.1, 0.6])

    # coeffs -> function
    func_from_coeffs = space.from_coefficient(original_coeffs)

    # function -> coeffs
    recovered_coeffs = space.to_coefficient(func_from_coeffs)

    print(f"Original coefficients: {original_coeffs}")
    print(f"Recovered coefficients: {recovered_coeffs}")
    print(f"Round-trip error: {np.linalg.norm(original_coeffs - recovered_coeffs)}")

    # Test 4: Test with callable function
    print("\n--- Test 4: to_coefficient with callable ---")

    def simple_func(x):
        return np.sin(2 * np.pi * x)

    coeffs_from_callable = space.to_coefficient(simple_func)
    print(f"Function: sin(2πx)")
    print(f"Coefficients: {coeffs_from_callable}")

    # Reconstruct and compare
    reconstructed = space.from_coefficient(coeffs_from_callable)
    test_x = np.linspace(0, 1, 10)
    original_vals = [simple_func(x) for x in test_x]
    reconstructed_vals = [reconstructed.evaluate(x) for x in test_x]

    print(f"Original values: {original_vals}")
    print(f"Reconstructed values: {reconstructed_vals}")
    print(f"Reconstruction error: {np.linalg.norm(np.array(original_vals) - np.array(reconstructed_vals))}")

    return True


def test_different_basis_types():
    """Test the new methods with different basis types."""

    print("\n=== Testing Different Basis Types ===\n")

    order = 1.5
    scale = 1.0
    dim = 3
    interval = (0, 1)

    for basis_type in ['fourier', 'sine', 'chebyshev']:
        print(f"--- Testing {basis_type.upper()} basis ---")

        space = Sobolev.create_standard_sobolev(
            order, scale, dim, interval=interval, basis_type=basis_type
        )

        # Test round-trip
        test_coeffs = np.array([1.0, 0.5, -0.3])

        func = space.from_coefficient(test_coeffs)
        recovered_coeffs = space.to_coefficient(func)

        print(f"Original: {test_coeffs}")
        print(f"Recovered: {recovered_coeffs}")
        print(f"Error: {np.linalg.norm(test_coeffs - recovered_coeffs)}")
        print(f"Function type: {type(func)}")
        print()


if __name__ == "__main__":
    try:
        test_basis_coefficient_methods()
        test_different_basis_types()
        print("✓ All tests passed!")
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
