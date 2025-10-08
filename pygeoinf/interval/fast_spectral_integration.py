"""
Fast transform methods for spectral integration in BesselSobolev operators.

This module provides optimized coefficient computation using fast transforms
instead of expensive numerical integration:

- DST for Dirichlet boundary conditions (sine eigenfunctions)
- DCT for Neumann boundary conditions (cosine eigenfunctions)
- DFT for periodic boundary conditions (Fourier eigenfunctions)

The key insight is that computing coefficients ∫ f(x) φₖ(x) dx for
orthogonal eigenfunctions can be done via fast transforms when f(x)
is sampled uniformly on the domain.
"""

import numpy as np
from scipy.fft import dst, dct, fft, ifft
from typing import Literal, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def fast_spectral_coefficients(
    f_samples: np.ndarray,
    boundary_condition: Literal['dirichlet', 'neumann', 'periodic'],
    domain_length: float = 1.0,
    n_coeffs: Optional[int] = None
) -> np.ndarray:
    """
    Compute spectral coefficients using fast transforms.

    This replaces expensive numerical integration ∫ f(x) φₖ(x) dx with
    fast transform methods that exploit the orthogonality of eigenfunctions.

    Args:
        f_samples: Uniform samples of f(x) on the domain
        boundary_condition: Type of boundary conditions
        domain_length: Length of the domain (b - a)
        n_coeffs: Number of coefficients to return (default: len(f_samples))

    Returns:
        np.ndarray: Spectral coefficients for the given basis

    Notes:
        - For Dirichlet: Uses DST-I (Discrete Sine Transform Type I)
        - For Neumann: Uses DCT-I (Discrete Cosine Transform Type I)
        - For Periodic: Uses DFT (Discrete Fourier Transform)
    """
    n_samples = len(f_samples)
    if n_coeffs is None:
        n_coeffs = n_samples

    if boundary_condition == 'dirichlet':
        return _fast_dirichlet_coefficients(f_samples, domain_length, n_coeffs)
    elif boundary_condition == 'neumann':
        return _fast_neumann_coefficients(f_samples, domain_length, n_coeffs)
    elif boundary_condition == 'periodic':
        return _fast_periodic_coefficients(f_samples, domain_length, n_coeffs)
    else:
        raise ValueError(f"Unsupported boundary condition: {boundary_condition}")


def _fast_dirichlet_coefficients(
    f_samples: np.ndarray,
    domain_length: float,
    n_coeffs: int
) -> np.ndarray:
    """
    Fast computation of Dirichlet coefficients using DST.

    For Dirichlet BCs, eigenfunctions are: φₖ(x) = √(2/L) sin(kπx/L)
    The DST directly computes coefficients: ∫ f(x) sin(kπx/L) dx
    """
    n_samples = len(f_samples)

    # Use DST Type I which matches our sine basis
    # DST-I: 2*Σⱼ f[j] sin(πjk/(N+1)) ≈ 2*(N+1) * ∫₀ᴸ f(x) sin(kπx/L) dx
    coeffs_raw = dst(f_samples, type=1)

    # Apply correct normalization factors
    # To get ∫₀ᴸ f(x) sin(kπx/L) dx from DST, divide by 2*(N+1)
    # Then multiply by √(2/L) for the normalized eigenfunction
    scaling = np.sqrt(2.0 / domain_length) / (2.0 * (n_samples + 1))
    coeffs_normalized = coeffs_raw * scaling

    # Return only the requested number of coefficients
    return coeffs_normalized[:n_coeffs]


def _fast_neumann_coefficients(
    f_samples: np.ndarray,
    domain_length: float,
    n_coeffs: int
) -> np.ndarray:
    """
    Fast computation of Neumann coefficients using DCT.

    For Neumann BCs, eigenfunctions are:
    - φ₀(x) = 1/√L (constant mode)
    - φₖ(x) = √(2/L) cos(kπx/L) for k ≥ 1
    """
    n_samples = len(f_samples)

    # Use DCT Type I which matches our cosine basis
    # DCT-I with N+1 points: DCT[k] ≈ 2*N * ∫₀ᴸ f(x) cos(kπx/L) dx
    coeffs_raw = dct(f_samples, type=1)

    # Apply correct normalization factors
    coeffs_normalized = np.zeros(n_coeffs)

    # For Neumann, we use N+1 sample points, so N = n_samples - 1
    N = n_samples - 1

    # Constant mode (k=0): c₀ = (1/√L) * ∫₀ᴸ f(x) dx
    if n_coeffs > 0:
        integral_0 = coeffs_raw[0] / (2.0 * N)  # type: ignore
        coeffs_normalized[0] = (1.0 / np.sqrt(domain_length)) * integral_0

    # Cosine modes (k≥1): cₖ = √(2/L) * ∫₀ᴸ f(x) cos(kπx/L) dx
    for k in range(1, min(n_coeffs, len(coeffs_raw))):
        integral_k = coeffs_raw[k] / (2.0 * N)  # type: ignore
        coeffs_normalized[k] = np.sqrt(2.0 / domain_length) * integral_k

    return coeffs_normalized


def _fast_periodic_coefficients(
    f_samples: np.ndarray,
    domain_length: float,
    n_coeffs: int
) -> np.ndarray:
    """
    Fast computation of periodic coefficients using DFT.

    For periodic BCs, eigenfunctions are: φₖ(x) = e^(2πikx/L) / √L
    The DFT directly computes these Fourier coefficients.
    """
    n_samples = len(f_samples)

    # Compute DFT
    coeffs_fft = fft(f_samples)

    # Apply normalization: DFT gives coefficients for e^(2πikx/N)
    # We want coefficients for e^(2πikx/L) / √L
    scaling = (domain_length / n_samples) / np.sqrt(domain_length)
    coeffs_normalized = coeffs_fft * scaling

    # For real functions, we typically want the real Fourier series
    # Convert complex coefficients to real ones
    coeffs_real = np.zeros(n_coeffs)

    # DC component (k=0)
    if n_coeffs > 0:
        coeffs_real[0] = coeffs_fft[0].real * scaling

    # Alternating cos/sin modes
    for k in range(1, min(n_coeffs, n_samples//2 + 1)):
        if 2*k-1 < n_coeffs:
            # Cosine coefficient: cos(2πkx/L)
            coeffs_real[2*k-1] = 2 * coeffs_fft[k].real * scaling
        if 2*k < n_coeffs:
            # Sine coefficient: sin(2πkx/L)
            coeffs_real[2*k] = -2 * coeffs_fft[k].imag * scaling

    return coeffs_real


def create_uniform_samples(
    func,
    domain: Tuple[float, float],
    n_samples: int,
    boundary_condition: Literal['dirichlet', 'neumann', 'periodic']
) -> np.ndarray:
    """
    Create uniform samples of a function for fast transform computation.

    Args:
        func: Function to sample (callable)
        domain: (a, b) domain endpoints
        n_samples: Number of sample points
        boundary_condition: Affects sampling strategy

    Returns:
        np.ndarray: Function samples at appropriate points
    """
    a, b = domain

    if boundary_condition == 'dirichlet':
        # For Dirichlet, exclude boundary points (where basis functions = 0)
        x = np.linspace(a, b, n_samples + 2)[1:-1]
    elif boundary_condition == 'neumann':
        # For Neumann, include boundary points
        x = np.linspace(a, b, n_samples)
    elif boundary_condition == 'periodic':
        # For periodic, exclude the right endpoint (periodic)
        x = np.linspace(a, b, n_samples + 1)[:-1]
    else:
        raise ValueError(f"Unsupported boundary condition: {boundary_condition}")

    return func(x)


# Benchmark and validation functions

def benchmark_integration_methods(
    func,
    domain: Tuple[float, float],
    boundary_condition: Literal['dirichlet', 'neumann', 'periodic'],
    n_coeffs: int = 100,
    n_samples: int = 1024
) -> dict:
    """
    Benchmark fast transforms vs numerical integration.

    Returns:
        dict: Timing and accuracy results
    """
    import time

    a, b = domain
    domain_length = b - a

    # Create function samples
    f_samples = create_uniform_samples(func, domain, n_samples, boundary_condition)

    # Fast transform method
    start_time = time.time()
    coeffs_fast = fast_spectral_coefficients(f_samples, boundary_condition, domain_length, n_coeffs)
    fast_time = time.time() - start_time

    # Note: For a full comparison, we'd need to implement the slow numerical integration
    # method here, but that would require the full function space infrastructure

    results = {
        'fast_transform_time': fast_time,
        'n_coeffs': n_coeffs,
        'n_samples': n_samples,
        'coefficients_computed': len(coeffs_fast),
        'method': boundary_condition,
    }

    logger.info(f"Fast transform ({boundary_condition}): {fast_time:.6f}s for {n_coeffs} coefficients")

    return results


if __name__ == "__main__":
    # Simple test/demo
    def test_func(x):
        return x * (1 - x)  # Simple test function

    domain = (0, 1)

    from typing import cast
    boundary_conditions = ['dirichlet', 'neumann', 'periodic']
    for bc in boundary_conditions:
        print(f"\\nTesting {bc} boundary conditions:")
        bc_typed = cast(Literal['dirichlet', 'neumann', 'periodic'], bc)
        results = benchmark_integration_methods(test_func, domain, bc_typed, n_coeffs=50, n_samples=512)
        print(f"  Computed {results['coefficients_computed']} coefficients in {results['fast_transform_time']:.6f}s")