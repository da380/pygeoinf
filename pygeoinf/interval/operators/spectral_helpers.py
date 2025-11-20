"""Helper utilities for spectral operator implementations.

This module provides common patterns used across spectral operators,
particularly for eigenfunction expansion and coefficient computation.
"""

from typing import Callable, List, Tuple, Optional
import numpy as np

from ..functions import Function


def build_eigenfunction_expansion(
    terms: List[Tuple[float, Function]],
    domain,
    codomain,
    tolerance: float = 1e-14
) -> Function:
    """
    Build a function from eigenfunction expansion with lazy evaluation.

    This pattern appears 8+ times across operators. It:
    1. Filters out negligible terms (|coeff| < tolerance)
    2. Creates a single callable that evaluates all terms efficiently
    3. Returns zero function if no significant terms

    Args:
        terms: List of (coefficient, eigenfunction) tuples
        domain: Domain space for the result (used for zero function)
        codomain: Codomain space for the result
        tolerance: Skip terms with |coeff| < tolerance (default: 1e-14)

    Returns:
        Function representing the eigenfunction expansion

    Example:
        >>> terms = [(0.5, phi_0), (0.3, phi_1), (1e-20, phi_2)]
        >>> f = build_eigenfunction_expansion(terms, domain, codomain)
        # Only first two terms used (third filtered out)
    """
    # Filter negligible terms
    significant_terms = [(c, ef) for c, ef in terms if abs(c) > tolerance]

    # Return zero if no significant terms
    if not significant_terms:
        return domain.zero

    # Create efficient evaluation callable
    def evaluate_expansion(x):
        result = (np.zeros_like(x, dtype=float)
                  if isinstance(x, np.ndarray) else 0.0)
        for coeff, eigfunc in significant_terms:
            result += coeff * eigfunc(x)
        return result

    return Function(codomain, evaluate_callable=evaluate_expansion)


def compute_spectral_coefficients_fast(
    operator,
    f: Function,
    coefficients: np.ndarray,
    scale_func: Optional[Callable[[int, float], float]] = None
) -> List[Tuple[float, Function]]:
    """
    Compute eigenfunction expansion terms using pre-computed coefficients.

    This is the "fast" path using DST/DCT/FFT transforms. Used in
    Laplacian, InverseLaplacian, and Bessel operators.

    Args:
        operator: Operator instance (needs get_eigenvalue, get_eigenfunction methods)
        f: Input function (not directly used, kept for API consistency)
        coefficients: Pre-computed spectral coefficients from fast transforms
        scale_func: Optional scaling function(index, eigenvalue) -> scale
                   If None, uses eigenvalue directly

    Returns:
        List of (scaled_coefficient, eigenfunction) tuples

    Example:
        >>> # For Laplacian: scale = eigenvalue
        >>> terms = compute_spectral_coefficients_fast(
        ...     lap_op, f, coeffs, scale_func=lambda i, ev: ev
        ... )
        >>>
        >>> # For Bessel: scale = (k² + λᵢ)^(s/2)
        >>> scale = lambda i, ev: (k**2 + ev)**(s/2)
        >>> terms = compute_spectral_coefficients_fast(
        ...     bessel_op, f, coeffs, scale_func=scale
        ... )
    """
    terms = []
    n_coeffs = len(coefficients)

    for i in range(n_coeffs):
        eigval = operator.get_eigenvalue(i)

        # Apply scaling
        if scale_func is not None:
            scale = scale_func(i, eigval)
        else:
            scale = eigval

        coeff = coefficients[i] * scale

        # Only append significant terms (filtering done in build_eigenfunction_expansion)
        eigfunc = operator.get_eigenfunction(i)
        terms.append((coeff, eigfunc))

    return terms


def compute_spectral_coefficients_slow(
    operator,
    f: Function,
    n_dofs: int,
    integration_method: str,
    integration_points: int,
    scale_func: Optional[Callable[[int, float], float]] = None,
    skip_zero_eigenvalues: bool = False
) -> List[Tuple[float, Function]]:
    """
    Compute eigenfunction expansion terms using numerical integration.

    This is the "slow" path for Robin BCs and other cases where fast
    transforms aren't available. Used in Laplacian, InverseLaplacian,
    and Bessel operators.

    Args:
        operator: Operator instance (needs get_eigenvalue, get_eigenfunction methods)
        f: Input function to project onto eigenfunctions
        n_dofs: Number of degrees of freedom (eigenfunctions to use)
        integration_method: Method for numerical integration ('simpson', etc.)
        integration_points: Number of points for integration
        scale_func: Optional scaling function(index, eigenvalue) -> scale
                   If None, uses eigenvalue directly
        skip_zero_eigenvalues: If True, skip terms with |eigenvalue| < 1e-14
                              (useful for InverseLaplacian)

    Returns:
        List of (scaled_coefficient, eigenfunction) tuples

    Example:
        >>> # For InverseLaplacian (skip zero eigenvalues)
        >>> terms = compute_spectral_coefficients_slow(
        ...     inv_lap, f, 100, 'simpson', 1000,
        ...     scale_func=lambda i, ev: ev,
        ...     skip_zero_eigenvalues=True
        ... )
    """
    terms = []

    for i in range(n_dofs):
        eigval = operator.get_eigenvalue(i)

        # Skip zero eigenvalues if requested
        if skip_zero_eigenvalues and abs(eigval) < 1e-14:
            continue

        eigfunc = operator.get_eigenfunction(i)

        # Compute coefficient via numerical integration
        coeff = (f * eigfunc).integrate(
            method=integration_method,
            n_points=integration_points
        )

        # Apply scaling
        if scale_func is not None:
            scale = scale_func(i, eigval)
        else:
            scale = eigval

        scaled_coeff = coeff * scale
        terms.append((scaled_coeff, eigfunc))

    return terms


def validate_eigenvalue(eigval: float, index: int, allow_negative: bool = True) -> None:
    """
    Validate an eigenvalue with descriptive error messages.

    Used in Bessel operators to ensure eigenvalues are valid for
    Bessel scaling operations.

    Args:
        eigval: The eigenvalue to validate
        index: The eigenfunction index (for error messages)
        allow_negative: Whether negative eigenvalues are allowed

    Raises:
        ValueError: If eigenvalue is None or invalid
    """
    if eigval is None:
        raise ValueError(f"Eigenvalue not available for index {index}")
    if not allow_negative and eigval < 0:
        raise ValueError(f"Negative eigenvalue {eigval} at index {index}")
