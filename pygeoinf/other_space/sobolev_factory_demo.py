#!/usr/bin/env python3
"""
Demonstration of the new Sobolev space factory with spectral inner products.
"""

import numpy as np
import matplotlib.pyplot as plt
from pygeoinf.other_space.interval_space import Sobolev

def demo_boundary_conditions():
    """Demonstrate different boundary conditions."""
    print("=== Boundary Conditions Demo ===")
    print()

    # Create spaces with different boundary conditions
    spaces = {
        'Periodic': Sobolev.factory(5, 1.0, interval=(0, 1),
                                  boundary_conditions={'type': 'periodic'}),
        'Dirichlet': Sobolev.factory(5, 1.0, interval=(0, 1),
                                   boundary_conditions={'type': 'dirichlet'}),
        'Neumann': Sobolev.factory(5, 1.0, interval=(0, 1),
                                 boundary_conditions={'type': 'neumann'})
    }

    for name, space in spaces.items():
        print(f"{name} BC:")
        print(f"  Basis: {[f.name for f in space.basis_functions]}")
        print(f"  Boundary conditions: {space.boundary_conditions}")
        print()


def demo_spectral_inner_product():
    """Demonstrate the spectral inner product."""
    print("=== Spectral Inner Product Demo ===")
    print()

    # Create spaces of different orders
    H1 = Sobolev.factory(5, 1.0, interval=(0, 1),
                        boundary_conditions={'type': 'periodic'})
    H2 = Sobolev.factory(5, 2.0, interval=(0, 1),
                        boundary_conditions={'type': 'periodic'})

    # Same coefficients in both spaces
    coeffs = np.array([1.0, 0.5, 0.3, 0.1, 0.05])
    f_h1 = H1.from_coefficient(coeffs)
    f_h2 = H2.from_coefficient(coeffs)

    print(f"Function coefficients: {coeffs}")
    print(f"H^1 norm squared: {H1.inner_product(f_h1, f_h1):.6f}")
    print(f"H^2 norm squared: {H2.inner_product(f_h2, f_h2):.6f}")
    print(f"Ratio H^2/H^1: {H2.inner_product(f_h2, f_h2) / H1.inner_product(f_h1, f_h1):.3f}")
    print()

    print("✓ Higher-order spaces have larger norms (more regularity)")
    print("✓ Spectral inner product: ⟨u,v⟩_H^s = ∑_k (1 + λ_k)^s û_k v̂_k")
    print("✓ No weak derivatives needed - uses eigenvalue scaling")
    print()


def demo_function_evaluation():
    """Demonstrate function evaluation."""
    print("=== Function Evaluation Demo ===")
    print()

    # Create a space
    space = Sobolev.factory(5, 2.0, interval=(0, 1),
                           boundary_conditions={'type': 'periodic'})

    # Create a function from coefficients
    coeffs = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
    f = space.from_coefficient(coeffs)

    print(f"Created function with coefficients: {coeffs}")
    print(f"Basis functions: {[bf.name for bf in space.basis_functions]}")
    print()

    # Evaluate at points
    x_vals = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    for x in x_vals:
        print(f"f({x:.2f}) = {f.evaluate(x):.6f}")
    print()


def demo_plotting():
    """Demonstrate plotting capabilities."""
    print("=== Plotting Demo ===")
    print()

    # Create spaces with different boundary conditions
    spaces = [
        ('Periodic', Sobolev.factory(5, 1.5, interval=(0, 1),
                                   boundary_conditions={'type': 'periodic'})),
        ('Dirichlet', Sobolev.factory(5, 1.5, interval=(0, 1),
                                    boundary_conditions={'type': 'dirichlet'})),
        ('Neumann', Sobolev.factory(5, 1.5, interval=(0, 1),
                                  boundary_conditions={'type': 'neumann'}))
    ]

    plt.figure(figsize=(15, 5))

    for i, (name, space) in enumerate(spaces):
        plt.subplot(1, 3, i+1)

        # Create a function with some coefficients
        coeffs = np.array([0.5, 0.8, 0.3, 0.4, 0.2])
        f = space.from_coefficient(coeffs)

        # Plot
        x = np.linspace(0, 1, 200)
        y = f.evaluate(x)
        plt.plot(x, y, linewidth=2, label=f'H^{space.order}')

        plt.title(f'{name} BC')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.grid(True, alpha=0.3)
        plt.legend()

    plt.tight_layout()
    plt.show()
    print("✓ Plotted functions with different boundary conditions")
    print()


def main():
    """Run all demonstrations."""
    print("Sobolev Space Factory Demonstration")
    print("=" * 50)
    print()

    demo_boundary_conditions()
    demo_spectral_inner_product()
    demo_function_evaluation()
    demo_plotting()

    print("=" * 50)
    print("All demonstrations completed successfully!")
    print()
    print("Key improvements:")
    print("- Only Laplacian eigenfunction bases (mathematically correct)")
    print("- Spectral inner product (no weak derivatives needed)")
    print("- Clear boundary condition handling")
    print("- Efficient coefficient-based representation")


if __name__ == "__main__":
    main()
