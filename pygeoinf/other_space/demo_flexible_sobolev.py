"""
Demonstration of the flexible Sobolev space design for basis-covariance alignment.

This script shows how users can provide their own basis functions that align
with their specific covariance operators for optimal Bayesian inversion.
"""

import numpy as np
import sys
import os

# Add the parent directory to the path to import our module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pygeoinf.pygeoinf.other_space.interval_space import Sobolev
from interval_domain import IntervalDomain, BoundaryConditions
from sobolev_functions import create_sobolev_function


def demo_custom_basis_covariance_alignment():
    """
    Demonstrate how users can create custom basis-covariance combinations.
    """

    print("=" * 70)
    print("Flexible Sobolev Space Design for Basis-Covariance Alignment")
    print("=" * 70)
    print()

    # Example 1: Fourier basis for frequency-based covariance
    print("1. Fourier Basis + Frequency-based Covariance")
    print("-" * 50)

    dim = 32
    interval = (0, 1)
    length = interval[1] - interval[0]

    # Define Fourier basis transformations
    def fourier_to_coeff(u):
        from scipy.fft import dct
        return dct(u, type=2, norm='ortho')

    def fourier_from_coeff(coeff):
        from scipy.fft import idct
        return idct(coeff, type=2, norm='ortho')

    # Define frequency-based Sobolev scaling
    def fourier_scaling(k):
        order = 1.5
        scale = 0.1
        freq = k * np.pi / length
        return (1 + (scale * freq) ** 2) ** order

    # Create Sobolev space
    fourier_space = Sobolev(
        dim, fourier_to_coeff, fourier_from_coeff, fourier_scaling,
        interval=interval
    )

    # Define covariance that aligns with Fourier basis (frequency decay)
    def frequency_covariance(k):
        freq = k * np.pi / length
        return np.exp(-0.1 * freq**2)

    # Create Gaussian measure
    gm_fourier = fourier_space.gaussian_measure(frequency_covariance)

    print(f"✓ Created Fourier-based Sobolev space (dim={dim})")
    print(f"✓ Covariance aligns with frequency structure")
    print(f"✓ Optimal for problems with frequency-based priors")
    print()

    # Example 2: Custom polynomial basis for algebraic covariance
    print("2. Custom Polynomial Basis + Algebraic Covariance")
    print("-" * 50)

    # Define a simple polynomial basis (for demonstration)
    def poly_to_coeff(u):
        # Simplified: just use the function values as "coefficients"
        # In practice, you'd implement proper polynomial projection
        return u.copy()

    def poly_from_coeff(coeff):
        return coeff.copy()

    # Define polynomial-degree based Sobolev scaling
    def poly_scaling(k):
        order = 2.0
        scale = 0.05
        return (1 + (scale * k) ** 2) ** order

    # Create Sobolev space
    poly_space = Sobolev(
        dim, poly_to_coeff, poly_from_coeff, poly_scaling,
        interval=interval
    )

    # Define covariance that aligns with polynomial basis (algebraic decay)
    def polynomial_covariance(k):
        return (k + 1) ** (-3)

    # Create Gaussian measure
    gm_poly = poly_space.gaussian_measure(polynomial_covariance)

    print(f"✓ Created polynomial-based Sobolev space (dim={dim})")
    print(f"✓ Covariance aligns with polynomial degree structure")
    print(f"✓ Optimal for problems with smooth, polynomial-like priors")
    print()

    # Example 3: Using the convenience factory for standard cases
    print("3. Standard Factory Method for Common Cases")
    print("-" * 50)

    # Use the factory method for standard Fourier basis
    standard_space = Sobolev.create_standard_sobolev(
        1.0, 0.1, dim, interval=interval, basis_type='fourier'
    )

    print(f"✓ Created standard Fourier space using factory method")
    print(f"✓ Convenient for common use cases")
    print()

    return {
        'fourier_space': fourier_space,
        'poly_space': poly_space,
        'standard_space': standard_space
    }


def demo_bayesian_inversion_context():
    """
    Show how this design facilitates Bayesian inversion.
    """

    print("4. Bayesian Inversion Context")
    print("-" * 50)

    print("The flexible design allows optimal alignment for:")
    print()
    print("• Prior knowledge:")
    print("  - Choose basis that naturally represents your prior")
    print("  - Align covariance with basis for computational efficiency")
    print()
    print("• Boundary conditions:")
    print("  - Fourier (cosine) for Neumann BC: ∂u/∂n = 0")
    print("  - Sine basis for Dirichlet BC: u = 0")
    print("  - Custom basis for Robin BC: αu + β∂u/∂n = 0")
    print()
    print("• Measurement operators:")
    print("  - Choose basis where measurement operator is simple")
    print("  - Point evaluations → basis with good pointwise properties")
    print("  - Integral measurements → basis with good integration properties")
    print()
    print("• Computational efficiency:")
    print("  - Diagonal covariance in chosen basis")
    print("  - Fast transforms (FFT, DCT, etc.)")
    print("  - Sparse representations")
    print()


def compare_with_fixed_discretization():
    """
    Compare with the previous fixed discretization approach.
    """

    print("5. Comparison with Fixed Discretization")
    print("-" * 50)

    print("Previous approach:")
    print("✗ Hard-coded basis choices (Chebyshev, Fourier, Sine)")
    print("✗ Assumed specific discretization")
    print("✗ Limited flexibility for custom applications")
    print()

    print("New flexible approach:")
    print("✓ User provides basis aligned with their covariance")
    print("✓ No assumptions about discretization")
    print("✓ Full control over basis-covariance alignment")
    print("✓ Follows HilbertSpace pattern")
    print("✓ Optimal for specific Bayesian inversion problems")
    print()


def demo_new_domain_structure():
    """
    Demonstrate the new mathematical domain structure.
    """
    print("=" * 70)
    print("New Mathematical Domain Structure")
    print("=" * 70)
    print()

    # Create mathematically rigorous interval domain
    domain = IntervalDomain(0, 1, boundary_type='closed', name="Unit Interval")
    print(f"✓ Created domain: {domain}")
    print(f"✓ Length: {domain.length}")
    print(f"✓ Center: {domain.center}")
    print()

    # Test domain operations
    test_points = np.array([-0.1, 0.0, 0.5, 1.0, 1.1])
    in_domain = domain.contains(test_points)
    print("Domain membership test:")
    for x, in_dom in zip(test_points, in_domain):
        print(f"  {x} ∈ {domain}: {in_dom}")
    print()

    # Integration capabilities
    print("Integration capabilities:")

    # Simple polynomial
    def f(x):
        return x**2

    integral_adaptive = domain.integrate(f, method='adaptive')
    integral_gauss = domain.integrate(f, method='gauss', n=10)
    analytical = 1/3  # ∫₀¹ x² dx = 1/3

    print(f"∫₀¹ x² dx:")
    print(f"  Adaptive: {integral_adaptive:.8f}")
    print(f"  Gauss-10: {integral_gauss:.8f}")
    print(f"  Analytical: {analytical:.8f}")
    print(f"  Error (adaptive): {abs(integral_adaptive - analytical):.2e}")
    print()

    # Boundary conditions
    print("Boundary condition examples:")
    bc_dirichlet = BoundaryConditions.dirichlet(0, 1)
    bc_neumann = BoundaryConditions.neumann(0, 0)
    bc_periodic = BoundaryConditions.periodic()

    print(f"  Dirichlet: u(0)=0, u(1)=1")
    print(f"  Neumann: u'(0)=0, u'(1)=0")
    print(f"  Periodic: u(0)=u(1), u'(0)=u'(1)")
    print()


def demo_sobolev_functions():
    """
    Demonstrate SobolevFunction capabilities.
    """
    print("=" * 70)
    print("Enhanced Sobolev Functions")
    print("=" * 70)
    print()

    # Create domain
    domain = IntervalDomain(0, 1)

    # Create a simple Sobolev function using Fourier basis
    n_modes = 8
    coefficients = np.zeros(n_modes)
    coefficients[0] = 1.0    # Constant term
    coefficients[1] = 0.5   # First cosine mode
    coefficients[2] = 0.25  # Second cosine mode

    f = create_sobolev_function(
        domain, coefficients, basis_type='fourier',
        sobolev_order=2.0, name="Test Function"
    )

    print(f"✓ Created {f}")

    # Function evaluation
    x_test = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    y_test = f.evaluate(x_test)
    print("\nFunction evaluation:")
    for x, y in zip(x_test, y_test):
        print(f"  f({x:.2f}) = {y:.4f}")
    print()

    # Integration
    integral = f.integrate()
    print(f"∫₀¹ f(x) dx = {integral:.6f}")

    # Norm
    norm_l2 = f.norm(order=0)  # L² norm
    norm_h1 = f.norm(order=1)  # H¹ norm
    print(f"‖f‖_{'{L²}'} = {norm_l2:.6f}")
    print(f"‖f‖_{'{H¹}'} = {norm_h1:.6f}")
    print()

    # Scalar operations
    g = 2.0 * f  # Scalar multiplication
    h = f + g    # Function addition
    print(f"✓ Created scaled function: 2f")
    print(f"✓ Created sum function: f + 2f = 3f")
    print(f"‖3f‖_{'{L²}'} = {h.norm(order=0):.6f} (should be 3 × {norm_l2:.6f})")
    print()


if __name__ == "__main__":
    try:
        # New enhanced demonstrations
        demo_new_domain_structure()
        demo_sobolev_functions()

        # Original demonstrations
        spaces = demo_custom_basis_covariance_alignment()
        demo_bayesian_inversion_context()
        compare_with_fixed_discretization()

        print("=" * 70)
        print("Key Benefits of the Enhanced Design:")
        print("=" * 70)
        print("1. Mathematically rigorous interval domains with topology/measure")
        print("2. Basis choice aligns with covariance operator")
        print("3. Functions know their domain structure")
        print("4. Integration and boundary conditions built-in")
        print("5. No assumptions about discretization")
        print("6. Optimal computational efficiency for specific problems")
        print("7. Follows the same pattern as base HilbertSpace class")
        print("8. User has full control over mathematical structure")

    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you're running from the correct directory")
        print("Note: New features require interval_domain.py and sobolev_functions.py")
    except Exception as e:
        print(f"Error running demonstration: {e}")
