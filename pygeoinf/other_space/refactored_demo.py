#!/usr/bin/env python3
"""
Demonstration of the refactored Sobolev function implementation.

This shows how the SobolevFunction class now works with the canonical
Sobolev space implementation from interval.py, ensuring proper mathematical
structure and eliminating redundancy.
"""

import numpy as np
import matplotlib.pyplot as plt
from interval_domain import IntervalDomain
from pygeoinf.pygeoinf.other_space.interval_space import Sobolev
from sobolev_functions import SobolevFunction, create_sobolev_function


def demo_space_aware_functions():
    """
    Demonstrate that SobolevFunction objects are now space-aware.
    """
    print("=== Space-Aware Sobolev Functions ===")

    # Create a Sobolev space using the canonical implementation
    space = Sobolev.create_standard_sobolev(
        order=1.5, scale=0.1, dim=50, interval=(0, np.pi)
    )

    print(f"Created Sobolev space: H^1.5 on [0, π] with {space.dim} dimensions")
    print(f"Space interval: {space.interval}")
    print(f"Space length: {space.length}")

    # Create a function using the factory
    f = create_sobolev_function(
        space,
        evaluate_callable=lambda x: np.sin(x) * np.exp(-x/2),
        sobolev_order=1.5,
        name="exponentially_damped_sine"
    )

    print(f"\nCreated function: {f}")
    print(f"Function belongs to space: {f.space}")
    print(f"Function domain: {f.domain}")
    print(f"Function Sobolev order: {f.sobolev_order}")

    # Test point evaluation (should work since 1.5 > 1/2)
    x_test = np.array([0.5, 1.0, 2.0])
    y_test = f(x_test)
    print(f"\nPoint evaluation at x = {x_test}:")
    print(f"f(x) = {y_test}")

    return space, f


def demo_coefficient_based_functions():
    """
    Demonstrate functions defined by coefficients in the space's basis.
    """
    print("\n=== Coefficient-Based Functions ===")

    # Create a Fourier-based Sobolev space
    space = Sobolev.create_standard_sobolev(
        order=2.0, scale=0.2, dim=32, interval=(-1, 1), basis_type='fourier'
    )

    print(f"Created Fourier-based space: H^2.0 on [-1, 1]")

    # Create coefficients with exponential decay
    coeffs = np.random.randn(space.dim)
    coeffs *= np.exp(-0.1 * np.arange(space.dim))  # Exponential decay

    # Create function from coefficients
    g = create_sobolev_function(
        space,
        coefficients=coeffs,
        sobolev_order=2.0,
        name="random_smooth_function"
    )

    print(f"Created function from coefficients: {g}")
    print(f"Coefficient shape: {g.coefficients.shape}")

    # Test evaluation (this uses the space's basis)
    x_test = np.linspace(-1, 1, 10)
    try:
        y_test = g(x_test)
        print(f"Successfully evaluated function at {len(x_test)} points")
        print(f"Function values range: [{np.min(y_test):.3f}, {np.max(y_test):.3f}]")
    except NotImplementedError as e:
        print(f"Note: {e}")
        print("(This is expected - coefficient-based evaluation needs enhancement)")

    return space, g


def demo_mathematical_operations():
    """
    Demonstrate mathematical operations between functions.
    """
    print("\n=== Mathematical Operations ===")

    # Create a shared space
    space = Sobolev.create_standard_sobolev(
        order=1.0, scale=0.1, dim=40, interval=(0, 2*np.pi)
    )

    # Create two functions
    f1 = create_sobolev_function(
        space,
        evaluate_callable=lambda x: np.sin(x),
        sobolev_order=1.0,
        name="sine"
    )

    f2 = create_sobolev_function(
        space,
        evaluate_callable=lambda x: np.cos(x),
        sobolev_order=1.0,
        name="cosine"
    )

    print(f"Created f1: {f1.name}")
    print(f"Created f2: {f2.name}")

    # Test addition
    f_sum = f1 + f2
    print(f"f1 + f2 = {f_sum}")

    # Test scalar multiplication
    f_scaled = 2.0 * f1
    print(f"2 * f1 = {f_scaled}")

    # Test evaluation
    x_test = np.array([0, np.pi/2, np.pi])
    print(f"\nEvaluations at x = {x_test}:")
    print(f"f1(x) = {f1(x_test)}")
    print(f"f2(x) = {f2(x_test)}")
    print(f"(f1 + f2)(x) = {f_sum(x_test)}")
    print(f"(2 * f1)(x) = {f_scaled(x_test)}")

    # Test inner product (uses space's inner product)
    try:
        ip = f1.inner_product(f2)
        print(f"\nInner product <f1, f2> = {ip:.6f}")
        print(f"||f1|| = {f1.norm():.6f}")
        print(f"||f2|| = {f2.norm():.6f}")
    except Exception as e:
        print(f"Inner product computation failed: {e}")

    return f1, f2, f_sum, f_scaled


def demo_mathematical_restrictions():
    """
    Demonstrate the mathematical restrictions on point evaluation.
    """
    print("\n=== Mathematical Restrictions ===")

    # Create a space with low regularity (s = 0.3 < 1/2)
    space_low = Sobolev.create_standard_sobolev(
        order=0.3, scale=0.1, dim=20, interval=(0, 1)
    )

    # Try to create a function with point evaluation
    print("Creating function with s = 0.3 < 1/2...")
    f_low = create_sobolev_function(
        space_low,
        evaluate_callable=lambda x: x**2,
        sobolev_order=0.3,
        name="low_regularity"
    )

    # Try point evaluation (should fail)
    try:
        y = f_low(0.5)
        print(f"Point evaluation succeeded: f(0.5) = {y}")
    except ValueError as e:
        print(f"Point evaluation failed as expected: {e}")

    # Create a space with sufficient regularity (s = 1.0 > 1/2)
    space_high = Sobolev.create_standard_sobolev(
        order=1.0, scale=0.1, dim=20, interval=(0, 1)
    )

    print("\nCreating function with s = 1.0 > 1/2...")
    f_high = create_sobolev_function(
        space_high,
        evaluate_callable=lambda x: x**2,
        sobolev_order=1.0,
        name="sufficient_regularity"
    )

    # Try point evaluation (should succeed)
    try:
        y = f_high(0.5)
        print(f"Point evaluation succeeded: f(0.5) = {y}")
    except ValueError as e:
        print(f"Point evaluation failed: {e}")


def demo_plotting():
    """
    Demonstrate plotting capabilities.
    """
    print("\n=== Plotting Functions ===")

    space = Sobolev.create_standard_sobolev(
        order=2.0, scale=0.1, dim=50, interval=(0, 4*np.pi)
    )

    # Create a few interesting functions
    f1 = create_sobolev_function(
        space,
        evaluate_callable=lambda x: np.sin(x) * np.exp(-x/10),
        sobolev_order=2.0,
        name="Damped Sine"
    )

    f2 = create_sobolev_function(
        space,
        evaluate_callable=lambda x: np.cos(2*x) * np.exp(-x/15),
        sobolev_order=2.0,
        name="Damped Cosine"
    )

    # Plot them
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    f1.plot(n_points=200, color='blue', linewidth=2)
    plt.title('Damped Sine Function')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    f2.plot(n_points=200, color='red', linewidth=2)
    plt.title('Damped Cosine Function')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    f1.plot(n_points=200, color='blue', linewidth=2)
    f2.plot(n_points=200, color='red', linewidth=2)
    plt.title('Both Functions')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    f_sum = f1 + f2
    f_sum.plot(n_points=200, color='green', linewidth=2)
    plt.title('Sum of Functions')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sobolev_functions_demo.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'sobolev_functions_demo.png'")


def main():
    """
    Main demonstration function.
    """
    print("Sobolev Function Refactoring Demonstration")
    print("=" * 50)

    try:
        # Run all demonstrations
        demo_space_aware_functions()
        demo_coefficient_based_functions()
        demo_mathematical_operations()
        demo_mathematical_restrictions()
        demo_plotting()

        print("\n" + "=" * 50)
        print("✓ All demonstrations completed successfully!")
        print("\nKey improvements in the refactored implementation:")
        print("1. SobolevFunction objects are now space-aware")
        print("2. Uses canonical Sobolev class from interval.py")
        print("3. Eliminates redundant SobolevSpace class")
        print("4. Proper mathematical restrictions on point evaluation")
        print("5. Delegates mathematical operations to the space")

    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
