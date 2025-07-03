#!/usr/bin/env python3
"""
Demo script showing how to use the create_standard_sobolev method
to create Sobolev spaces with basis functions as SobolevFunction instances.
"""

import numpy as np
import matplotlib.pyplot as plt
from pygeoinf.other_space.interval_space import Sobolev


def demo_sobolev_functions():
    """Demonstrate the SobolevFunction basis functions."""

    print("=== Demo: Sobolev Space with SobolevFunction Basis ===\n")

    # Parameters
    order = 1.5  # s > 0.5 for point evaluation
    scale = 1.0
    dim = 4
    interval = (0, 1)

    print(f"Creating Sobolev space H^{order}([{interval[0]}, {interval[1]}])")
    print(f"Dimension: {dim}")
    print(f"Scale parameter: {scale}")

    # Create space with different basis types
    for basis_type in ['fourier', 'sine', 'chebyshev']:
        print(f"\n--- {basis_type.upper()} BASIS ---")

        # Create the space
        space = Sobolev.create_standard_sobolev(
            order, scale, dim, interval=interval, basis_type=basis_type
        )

        # Get basis functions
        basis_functions = space.basis_functions

        print(f"Created {len(basis_functions)} basis functions:")

        # Evaluate and display basis functions
        x_eval = np.linspace(interval[0], interval[1], 100)

        for i, func in enumerate(basis_functions):
            print(f"  {i}: {func.name}")

            # Show some properties
            print(f"      Sobolev order: {func.sobolev_order}")
            print(f"      Domain: {func.domain}")

            # Evaluate at a few points
            test_points = [0.25, 0.5, 0.75]
            values = [func.evaluate(x) for x in test_points]
            print(f"      Values at {test_points}: {values}")

            # You can also use the function call syntax
            alt_values = [func(x) for x in test_points]
            print(f"      Same values using f(x): {alt_values}")

        print(f"\nSpace properties:")
        print(f"  Dimension: {space.dim}")
        print(f"  Interval: {space.interval}")
        print(f"  Order: {space.order}")
        print(f"  Length: {space.length}")

        # Show how to work with coefficient representations
        print(f"\n  Converting between function values and coefficients:")

        # Create some test function values
        test_values = np.random.randn(dim)
        print(f"  Test values: {test_values}")

        # Convert to coefficients
        coeffs = space.to_coefficient(test_values)
        print(f"  Coefficients: {coeffs}")

        # Convert back to function values
        reconstructed = space.from_coefficient(coeffs)
        print(f"  Reconstructed: {reconstructed}")
        print(f"  Reconstruction error: {np.linalg.norm(test_values - reconstructed)}")


def demo_sobolev_function_properties():
    """Demonstrate specific properties of SobolevFunction."""

    print("\n=== Demo: SobolevFunction Properties ===\n")

    # Create a simple space
    space = Sobolev.create_standard_sobolev(
        1.0, 1.0, 3, interval=(0, 1), basis_type='fourier'
    )

    # Get the first basis function
    func = space.basis_functions[0]

    print(f"Function: {func.name}")
    print(f"Space: {func.space}")
    print(f"Domain: {func.domain}")
    print(f"Sobolev order: {func.sobolev_order}")

    # Test evaluation
    x_test = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    values = func.evaluate(x_test)
    print(f"\nEvaluation at {x_test}:")
    print(f"Values: {values}")

    # Test domain checking
    print(f"\nTesting domain checking:")
    try:
        # This should work
        val_in_domain = func.evaluate(0.5, check_domain=True)
        print(f"  f(0.5) = {val_in_domain} (in domain)")
    except Exception as e:
        print(f"  Error: {e}")

    try:
        # This should fail
        val_out_domain = func.evaluate(1.5, check_domain=True)
        print(f"  f(1.5) = {val_out_domain} (out of domain)")
    except Exception as e:
        print(f"  Error: {e}")


if __name__ == "__main__":
    demo_sobolev_functions()
    demo_sobolev_function_properties()
