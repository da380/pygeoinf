"""
Demonstration of basis-covariance alignment for interval Sobolev spaces.

This script shows how different basis choices (Chebyshev, Fourier, Sine)
align with different covariance structures for optimal Bayesian inversion.
"""

import numpy as np
import matplotlib.pyplot as plt
from interval import Sobolev


def demo_basis_alignment():
    """
    Demonstrate how different bases affect the covariance structure.
    """

    # Common parameters
    n_points = 64
    order = 1.0
    scale = 0.1
    interval = (0, 1)

    # Create spaces with different bases
    spaces = {
        'Chebyshev': Sobolev(n_points, order, scale, interval=interval,
                            basis_type='chebyshev'),
        'Fourier (Cosine)': Sobolev(n_points, order, scale, interval=interval,
                                   basis_type='fourier'),
        'Sine': Sobolev(n_points, order, scale, interval=interval,
                       basis_type='sine')
    }

    # Different covariance functions for demonstration
    # These represent different prior assumptions about the function

    def polynomial_covariance(k):
        """Good for Chebyshev basis - polynomial decay"""
        return np.exp(-0.5 * k)

    def frequency_covariance(k):
        """Good for Fourier/Sine basis - frequency decay"""
        freq = k * np.pi
        return np.exp(-0.1 * freq**2)

    def exponential_covariance(k):
        """Exponential decay - works reasonably with all bases"""
        return np.exp(-0.2 * k)

    covariances = {
        'Polynomial': polynomial_covariance,
        'Frequency': frequency_covariance,
        'Exponential': exponential_covariance
    }

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Basis-Covariance Alignment for Interval Sobolev Spaces')

    for i, (cov_name, cov_func) in enumerate(covariances.items()):
        for j, (basis_name, space) in enumerate(spaces.items()):
            ax = axes[i, j]

            # Create Gaussian measure with this covariance
            gm = space.gaussian_measure(cov_func)

            # Sample some functions
            samples = []
            for _ in range(5):
                sample = gm.sample()
                samples.append(sample)

            # Plot the samples
            x = space.points()
            for sample in samples:
                ax.plot(x, sample, alpha=0.7, linewidth=1)

            ax.set_title(f'{basis_name} + {cov_name}')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(interval)

            if j == 0:
                ax.set_ylabel('Function Value')
            if i == 2:
                ax.set_xlabel('x')

    plt.tight_layout()
    plt.savefig('basis_covariance_alignment.png', dpi=150, bbox_inches='tight')
    plt.show()

    return spaces, covariances


def analyze_computational_efficiency():
    """
    Analyze computational properties of different basis-covariance combinations.
    """
    print("=== Computational Efficiency Analysis ===\n")

    n_points = 64
    order = 1.0
    scale = 0.1

    # Test how diagonal the covariance matrix is in different bases
    def measure_diagonality(space, cov_func):
        """Measure how diagonal the covariance is in the chosen basis"""
        gm = space.gaussian_measure(cov_func)

        # Extract covariance matrix (this is conceptual - actual implementation
        # uses covariance factor)
        cov_values = np.array([cov_func(k) for k in range(space.dim)])

        # For this demo, we'll look at the decay rate of coefficients
        # Ideally, we want fast decay for computational efficiency
        decay_rate = np.mean(np.diff(np.log(cov_values + 1e-10)))

        return decay_rate, cov_values

    bases = ['chebyshev', 'fourier', 'sine']

    # Polynomial covariance (should work best with Chebyshev)
    def poly_cov(k): return (k + 1)**(-2)

    # Frequency-based covariance (should work best with Fourier/Sine)
    def freq_cov(k): return np.exp(-0.1 * (k * np.pi)**2)

    print("1. Polynomial Covariance (polynomial decay):")
    for basis in bases:
        space = Sobolev(n_points, order, scale, basis_type=basis)
        decay_rate, values = measure_diagonality(space, poly_cov)
        print(f"   {basis:10s}: decay rate = {decay_rate:.3f}")

    print("\n2. Frequency Covariance (frequency-based decay):")
    for basis in bases:
        space = Sobolev(n_points, order, scale, basis_type=basis)
        decay_rate, values = measure_diagonality(space, freq_cov)
        print(f"   {basis:10s}: decay rate = {decay_rate:.3f}")


def demo_bayesian_inversion_context():
    """
    Show how basis choice affects Bayesian inversion setup.
    """
    print("\n=== Bayesian Inversion Context ===\n")

    print("In Bayesian inversion, we typically have:")
    print("1. Prior knowledge about the function (encoded in covariance)")
    print("2. Measurement operator (how we observe the function)")
    print("3. Noise model")
    print()
    print("The basis choice should align with:")
    print("- The covariance structure (for efficient representation)")
    print("- The measurement operator (for efficient forward computation)")
    print("- Boundary conditions of the problem")
    print()

    # Example scenarios
    scenarios = {
        'Smooth functions with polynomial structure': {
            'best_basis': 'chebyshev',
            'covariance': 'polynomial decay',
            'reason': 'Chebyshev polynomials naturally represent smooth functions'
        },
        'Functions from differential equations (Neumann BC)': {
            'best_basis': 'fourier',
            'covariance': 'frequency-based (from Green\'s function)',
            'reason': 'Cosine basis satisfies Neumann boundary conditions'
        },
        'Functions from differential equations (Dirichlet BC)': {
            'best_basis': 'sine',
            'covariance': 'frequency-based (from Green\'s function)',
            'reason': 'Sine basis satisfies Dirichlet boundary conditions'
        }
    }

    for scenario, details in scenarios.items():
        print(f"Scenario: {scenario}")
        print(f"  Best basis: {details['best_basis']}")
        print(f"  Covariance: {details['covariance']}")
        print(f"  Reason: {details['reason']}")
        print()


if __name__ == "__main__":
    print("Demonstrating Basis-Covariance Alignment in Interval Sobolev Spaces")
    print("="*70)

    # Run demonstrations
    demo_bayesian_inversion_context()
    analyze_computational_efficiency()

    print("\nGenerating visualization...")
    spaces, covariances = demo_basis_alignment()

    print("\nKey takeaways:")
    print("1. Different bases are optimal for different covariance structures")
    print("2. Chebyshev: Good for polynomial/algebraic decay")
    print("3. Fourier (Cosine): Good for frequency-based decay, Neumann BC")
    print("4. Sine: Good for frequency-based decay, Dirichlet BC")
    print("5. The choice should match your prior knowledge and problem structure")
