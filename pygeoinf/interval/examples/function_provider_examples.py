"""
Example usage of the generalized function provider system.

This demonstrates how to create various function families and use them
for different purposes.
"""

import numpy as np
import matplotlib.pyplot as plt
from function_providers import (
    DiscontinuousFunctionProvider,
    SplineFunctionProvider,
    WaveletFunctionProvider,
    create_fourier_provider
)
from enhanced_providers import (
    GeneralizedBasisProvider,
    create_spline_basis_provider
)


def example_random_discontinuous_functions():
    """Example: Generate random discontinuous functions."""
    domain = (0, 1)

    # Create provider for discontinuous functions
    disc_provider = DiscontinuousFunctionProvider(domain, random_state=42)

    # Generate some random functions
    print("Generating random discontinuous functions...")
    for i in range(3):
        func = disc_provider.get_function(n_discontinuities=2)
        print(f"Function {i+1}: {func.name}")

        # Evaluate on a grid
        x = np.linspace(0, 1, 100)
        y = func(x)

        plt.subplot(1, 3, i+1)
        plt.plot(x, y, label=func.name)
        plt.title(f"Random Discontinuous {i+1}")
        plt.legend()

    plt.tight_layout()
    plt.show()


def example_spline_family():
    """Example: Generate splines with different parameters."""
    domain = (0, 1)

    # Create spline provider
    spline_provider = SplineFunctionProvider(domain)

    # Different spline configurations
    configs = [
        {'degree': 1, 'knots': np.linspace(0, 1, 4),
         'coefficients': [1, -1, 0.5]},
        {'degree': 2, 'knots': np.linspace(0, 1, 5),
         'coefficients': [0, 1, -0.5, 0.2]},
        {'degree': 3, 'knots': np.linspace(0, 1, 6),
         'coefficients': [1, 0, -1, 1, 0.5]}
    ]

    print("Generating spline functions with different parameters...")
    x = np.linspace(0, 1, 100)

    for i, params in enumerate(configs):
        func = spline_provider.get_function_by_parameters(params)
        y = func(x)

        plt.subplot(1, 3, i+1)
        plt.plot(x, y, label=f"Degree {params['degree']}")
        plt.title(f"Spline Degree {params['degree']}")
        plt.legend()

    plt.tight_layout()
    plt.show()


def example_mixed_function_families():
    """Example: Compare different function families."""
    domain = (0, 1)
    x = np.linspace(0, 1, 200)

    # Create different providers
    fourier_provider = create_fourier_provider(domain)
    wavelet_provider = WaveletFunctionProvider(domain)
    disc_provider = DiscontinuousFunctionProvider(domain, random_state=123)

    # Get sample functions
    fourier_func = fourier_provider.get_function_by_index(3)  # Some Fourier mode
    wavelet_func = wavelet_provider.get_function_by_index(2, level=1)  # Wavelet
    disc_func = disc_provider.get_function(n_discontinuities=1)  # Discontinuous

    functions = [
        (fourier_func, "Fourier"),
        (wavelet_func, "Wavelet"),
        (disc_func, "Discontinuous")
    ]

    print("Comparing different function families...")
    for i, (func, name) in enumerate(functions):
        y = func(x)

        plt.subplot(1, 3, i+1)
        plt.plot(x, y)
        plt.title(name)
        plt.grid(True)

    plt.tight_layout()
    plt.show()


def example_basis_provider_usage():
    """Example: Using function providers as basis providers."""

    # This would work if you have a space object
    # Here's the conceptual usage:

    class MockSpace:
        def __init__(self, dim, domain):
            self.dim = dim
            self.function_domain = type('Domain', (), {'a': domain[0], 'b': domain[1]})()

    # Create a mock space
    space = MockSpace(dim=5, domain=(0, 1))

    # Method 1: Direct use of GeneralizedBasisProvider
    fourier_provider = create_fourier_provider((0, 1), space)
    basis_provider = GeneralizedBasisProvider(space, fourier_provider)

    print("Using Fourier functions as basis:")
    for i in range(3):
        func = basis_provider.get_basis_function(i)
        print(f"  Basis function {i}: {func.name}")

    # Method 2: Factory function for splines
    spline_basis = create_spline_basis_provider(space, degree=2)

    print("Using spline functions as basis:")
    for i in range(min(3, space.dim)):
        func = spline_basis.get_basis_function(i)
        print(f"  Basis function {i}: {func.name}")


def example_extensibility():
    """Example: Creating custom function providers."""

    from function_providers import ParametricFunctionProvider

    class GaussianBumpProvider(ParametricFunctionProvider):
        """Custom provider for Gaussian bump functions."""

        def __init__(self, domain, space=None):
            self.domain = domain
            self.space = space

        def get_default_parameters(self):
            return {'center': 0.5, 'width': 0.1, 'amplitude': 1.0}

        def get_function_by_parameters(self, parameters, **kwargs):
            from function_providers import Function  # Would need proper import

            center = parameters['center']
            width = parameters['width']
            amplitude = parameters['amplitude']

            def gaussian_bump(x):
                return amplitude * np.exp(-((x - center) / width)**2)

            # Would return proper Function object
            return type('MockFunction', (), {
                'name': f'gaussian_c{center}_w{width}_a{amplitude}',
                '__call__': lambda self, x: gaussian_bump(x)
            })()

    # Usage
    domain = (0, 1)
    gaussian_provider = GaussianBumpProvider(domain)

    # Generate Gaussians with different parameters
    params_list = [
        {'center': 0.2, 'width': 0.05, 'amplitude': 2.0},
        {'center': 0.5, 'width': 0.1, 'amplitude': 1.0},
        {'center': 0.8, 'width': 0.15, 'amplitude': 1.5}
    ]

    print("Custom Gaussian bump functions:")
    for params in params_list:
        func = gaussian_provider.get_function_by_parameters(params)
        print(f"  {func.name}")


if __name__ == "__main__":
    print("=== Generalized Function Provider Examples ===\n")

    print("1. Random discontinuous functions")
    # example_random_discontinuous_functions()

    print("\n2. Parametric spline functions")
    # example_spline_family()

    print("\n3. Mixed function families")
    # example_mixed_function_families()

    print("\n4. Basis provider usage")
    example_basis_provider_usage()

    print("\n5. Custom providers")
    example_extensibility()
