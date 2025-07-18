"""
DOLFINx-Pygeoinf Bridge Classes

This module provides bridge classes to convert between pygeoinf's Sobolev spaces
and DOLFINx's finite element spaces, enabling seamless integration between the
mathematical framework (pygeoinf) and computational engine (DOLFINx).

Classes:
    DOLFINxSobolevBridge: Bridge between pygeoinf Sobolev spaces and DOLFINx function spaces

Author: Adrian-Mag
Date: July 2025
"""

import numpy as np
from typing import Optional, Union, Callable
from scipy.integrate import quad

try:
    import dolfinx
    import ufl
    from dolfinx import fem
    DOLFINX_AVAILABLE = True
except ImportError:
    DOLFINX_AVAILABLE = False
    # Create dummy classes for type hints when DOLFINx is not available
    class fem:
        class Function:
            pass
    class dolfinx:
        class mesh:
            class Mesh:
                pass

from .interval_space import Sobolev
from .sobolev_functions import SobolevFunction


class DOLFINxSobolevBridge:
    """
    Bridge between pygeoinf Sobolev spaces and DOLFINx function spaces.

    This class provides methods to convert between:
    - pygeoinf SobolevFunction objects and DOLFINx Function objects
    - Coefficient vectors and DOLFINx Function objects
    - Handles basis function reconstruction and projection

    The bridge respects the mathematical structure of both frameworks while
    providing efficient conversion routines.
    """

    def __init__(self, sobolev_space: Sobolev, dolfinx_domain, dolfinx_function_space):
        """
        Initialize the bridge between pygeoinf and DOLFINx.

        Args:
            sobolev_space: pygeoinf Sobolev space
            dolfinx_domain: DOLFINx mesh/domain
            dolfinx_function_space: DOLFINx function space

        Raises:
            ImportError: If DOLFINx is not available
            ValueError: If dimensions are incompatible
        """
        if not DOLFINX_AVAILABLE:
            raise ImportError("DOLFINx is not available. Install DOLFINx to use this bridge.")

        self.sobolev_space = sobolev_space
        self.dolfinx_domain = dolfinx_domain
        self.dolfinx_function_space = dolfinx_function_space

        # Get spatial coordinates for evaluation
        self.x_coords = dolfinx_domain.geometry.x[:, 0]  # Assuming 1D for now

        # Check dimension compatibility
        sobolev_dim = self.sobolev_space.dim
        dolfinx_dim = self.dolfinx_function_space.dofmap.index_map.size_global

        if sobolev_dim != dolfinx_dim:
            print(f"Warning: Dimension mismatch - Sobolev: {sobolev_dim}, DOLFINx: {dolfinx_dim}")
            print("This may affect accuracy of conversions.")

    def sobolev_to_dolfinx(self, sobolev_function: SobolevFunction) -> 'fem.Function':
        """
        Convert pygeoinf SobolevFunction to DOLFINx Function.

        Args:
            sobolev_function: SobolevFunction to convert

        Returns:
            DOLFINx Function object

        Raises:
            ValueError: If SobolevFunction cannot be evaluated
        """
        # Evaluate the Sobolev function at DOLFINx nodes
        if hasattr(sobolev_function, 'evaluate_callable') and sobolev_function.evaluate_callable is not None:
            # Function defined by callable
            values = sobolev_function.evaluate_callable(self.x_coords)
        elif hasattr(sobolev_function, 'coefficients') and sobolev_function.coefficients is not None:
            # Function defined by coefficients - reconstruct from basis
            values = self._reconstruct_from_coefficients(sobolev_function.coefficients)
        else:
            raise ValueError("SobolevFunction must have either evaluate_callable or coefficients")

        # Create DOLFINx function
        dolfinx_func = fem.Function(self.dolfinx_function_space)
        dolfinx_func.x.array[:] = values.real  # Ensure real values
        return dolfinx_func

    def dolfinx_to_sobolev(self, dolfinx_function: 'fem.Function', name: Optional[str] = None) -> SobolevFunction:
        """
        Convert DOLFINx Function to pygeoinf SobolevFunction.

        Args:
            dolfinx_function: DOLFINx Function to convert
            name: Optional name for the resulting SobolevFunction

        Returns:
            SobolevFunction object
        """
        # Get function values at nodes
        values = dolfinx_function.x.array.real.copy()

        # Project onto Sobolev space basis to get coefficients
        coefficients = self._project_to_sobolev_basis(values)

        # Create SobolevFunction
        return SobolevFunction(
            self.sobolev_space,
            coefficients=coefficients,
            name=name or 'dolfinx_converted'
        )

    def coefficients_to_dolfinx(self, coefficients: np.ndarray) -> 'fem.Function':
        """
        Convert coefficient vector to DOLFINx Function.

        Args:
            coefficients: Coefficient vector in Sobolev basis

        Returns:
            DOLFINx Function object
        """
        # Reconstruct function values from coefficients
        values = self._reconstruct_from_coefficients(coefficients)

        # Create DOLFINx function
        dolfinx_func = fem.Function(self.dolfinx_function_space)
        dolfinx_func.x.array[:] = values.real
        return dolfinx_func

    def dolfinx_to_coefficients(self, dolfinx_function: 'fem.Function') -> np.ndarray:
        """
        Convert DOLFINx Function to coefficient vector.

        Args:
            dolfinx_function: DOLFINx Function to convert

        Returns:
            Coefficient vector in Sobolev basis
        """
        values = dolfinx_function.x.array.real.copy()
        return self._project_to_sobolev_basis(values)

    def _reconstruct_from_coefficients(self, coefficients: np.ndarray) -> np.ndarray:
        """
        Reconstruct function values from Sobolev basis coefficients.

        Args:
            coefficients: Coefficients in Sobolev basis

        Returns:
            Function values at DOLFINx nodes
        """
        if not hasattr(self.sobolev_space, '_basis_functions') or self.sobolev_space._basis_functions is None:
            # Simple case: coefficients are the function values (e.g., nodal basis)
            # Interpolate/resize if dimensions don't match
            if len(coefficients) == len(self.x_coords):
                return coefficients
            else:
                # Simple linear interpolation for dimension mismatch
                return np.interp(self.x_coords,
                               np.linspace(self.sobolev_space._a, self.sobolev_space._b, len(coefficients)),
                               coefficients)

        # Reconstruct using basis functions
        values = np.zeros_like(self.x_coords)
        for i, coeff in enumerate(coefficients):
            if i < len(self.sobolev_space._basis_functions):
                basis_func = self.sobolev_space._basis_functions[i]
                if hasattr(basis_func, 'evaluate_callable') and basis_func.evaluate_callable is not None:
                    basis_values = basis_func.evaluate_callable(self.x_coords)
                    values += coeff * basis_values
                elif callable(basis_func):
                    # Direct callable basis function
                    values += coeff * basis_func(self.x_coords)

        return values

    def _project_to_sobolev_basis(self, values: np.ndarray) -> np.ndarray:
        """
        Project function values onto Sobolev basis to get coefficients.

        Args:
            values: Function values at DOLFINx nodes

        Returns:
            Coefficients in Sobolev basis
        """
        if not hasattr(self.sobolev_space, '_basis_functions') or self.sobolev_space._basis_functions is None:
            # Simple case: values are the coefficients (e.g., nodal basis)
            if len(values) == self.sobolev_space.dim:
                return values
            else:
                # Resize/interpolate if dimensions don't match
                sobolev_points = np.linspace(self.sobolev_space._a, self.sobolev_space._b, self.sobolev_space.dim)
                return np.interp(sobolev_points, self.x_coords, values)

        # Project using inner product with basis functions
        coefficients = np.zeros(self.sobolev_space.dim)
        dx = (self.sobolev_space._b - self.sobolev_space._a) / (len(self.x_coords) - 1)

        for i, basis_func in enumerate(self.sobolev_space._basis_functions):
            if i >= len(coefficients):
                break

            if hasattr(basis_func, 'evaluate_callable') and basis_func.evaluate_callable is not None:
                basis_values = basis_func.evaluate_callable(self.x_coords)
            elif callable(basis_func):
                basis_values = basis_func(self.x_coords)
            else:
                continue

            # Compute inner product using trapezoidal rule
            # For L2 projection: (f, φ_i) = ∫ f(x) φ_i(x) dx
            coefficients[i] = np.trapz(values * basis_values, dx=dx)

        return coefficients

    def test_round_trip(self, test_function: Union[SobolevFunction, Callable], name: str = "test") -> dict:
        """
        Test round-trip conversion: Sobolev -> DOLFINx -> Sobolev.

        Args:
            test_function: Either a SobolevFunction or callable to test
            name: Name for the test

        Returns:
            Dictionary with error metrics
        """
        if callable(test_function):
            # Create SobolevFunction from callable
            test_sobolev = SobolevFunction(
                self.sobolev_space,
                evaluate_callable=test_function,
                name=name
            )
        else:
            test_sobolev = test_function

        # Round trip: Sobolev -> DOLFINx -> Sobolev
        dolfinx_func = self.sobolev_to_dolfinx(test_sobolev)
        recovered_sobolev = self.dolfinx_to_sobolev(dolfinx_func, name=f"{name}_recovered")

        # Compare original and recovered
        if hasattr(test_sobolev, 'coefficients') and test_sobolev.coefficients is not None:
            original_coeffs = test_sobolev.coefficients
        else:
            # Project original function to get coefficients
            if hasattr(test_sobolev, 'evaluate_callable'):
                original_values = test_sobolev.evaluate_callable(self.x_coords)
                original_coeffs = self._project_to_sobolev_basis(original_values)
            else:
                raise ValueError("Cannot extract coefficients from test function")

        recovered_coeffs = recovered_sobolev.coefficients

        # Compute errors
        l2_error = np.linalg.norm(recovered_coeffs - original_coeffs)
        rel_error = l2_error / np.linalg.norm(original_coeffs) if np.linalg.norm(original_coeffs) > 0 else l2_error
        max_error = np.max(np.abs(recovered_coeffs - original_coeffs))

        return {
            'l2_error': l2_error,
            'relative_error': rel_error,
            'max_error': max_error,
            'original_coefficients': original_coeffs,
            'recovered_coefficients': recovered_coeffs,
            'original_function': test_sobolev,
            'recovered_function': recovered_sobolev
        }


class DOLFINxBridgeFactory:
    """
    Factory class to create DOLFINx bridges with appropriate settings.
    """

    @staticmethod
    def create_interval_bridge(sobolev_space: Sobolev,
                             n_elements: int = 32,
                             element_degree: int = 1) -> DOLFINxSobolevBridge:
        """
        Create a bridge for 1D interval problems.

        Args:
            sobolev_space: Sobolev space on interval
            n_elements: Number of finite elements
            element_degree: Polynomial degree of elements

        Returns:
            DOLFINxSobolevBridge for the interval

        Raises:
            ImportError: If DOLFINx is not available
        """
        if not DOLFINX_AVAILABLE:
            raise ImportError("DOLFINx is not available. Install DOLFINx to use this bridge.")

        from mpi4py import MPI
        from dolfinx import mesh

        # Get interval from Sobolev space
        interval_start = getattr(sobolev_space, '_a', 0.0)
        interval_end = getattr(sobolev_space, '_b', 1.0)

        # Create DOLFINx domain and function space
        dolfinx_domain = mesh.create_interval(MPI.COMM_WORLD, n_elements, [interval_start, interval_end])
        dolfinx_V = fem.functionspace(dolfinx_domain, ("Lagrange", element_degree))

        return DOLFINxSobolevBridge(sobolev_space, dolfinx_domain, dolfinx_V)

    @staticmethod
    def create_rectangle_bridge(sobolev_space,
                              nx: int = 32, ny: int = 32,
                              element_degree: int = 1) -> DOLFINxSobolevBridge:
        """
        Create a bridge for 2D rectangular problems.

        Args:
            sobolev_space: Sobolev space on rectangle
            nx, ny: Number of elements in x and y directions
            element_degree: Polynomial degree of elements

        Returns:
            DOLFINxSobolevBridge for the rectangle

        Note:
            This is a placeholder for future 2D implementation
        """
        raise NotImplementedError("2D rectangle bridge not yet implemented")


def demo_bridge():
    """
    Demonstration of the DOLFINx-Sobolev bridge functionality.
    """
    if not DOLFINX_AVAILABLE:
        print("DOLFINx not available - cannot run demo")
        return

    print("DOLFINx-Sobolev Bridge Demo")
    print("=" * 40)

    # Create a Sobolev space
    from .interval_space import Sobolev

    interval = (0.0, 1.0)
    sobolev_dim = 16

    try:
        sobolev_space = Sobolev.create_standard_sobolev(
            sobolev_dim, 1.0,  # H^1
            interval=interval,
            basis_type='fourier',
            boundary_conditions={'type': 'dirichlet'}
        )

        # Create bridge
        bridge = DOLFINxBridgeFactory.create_interval_bridge(sobolev_space, n_elements=32)

        # Test with a simple function
        def test_func(x):
            return np.sin(np.pi * x)

        print(f"Testing bridge with sin(πx)...")
        result = bridge.test_round_trip(test_func, "sin_pi_x")

        print(f"L2 error: {result['l2_error']:.2e}")
        print(f"Relative error: {result['relative_error']:.2e}")
        print(f"Max error: {result['max_error']:.2e}")

        if result['relative_error'] < 1e-10:
            print("✓ Bridge test successful!")
        else:
            print("! Bridge test shows some error (may be due to basis mismatch)")

    except Exception as e:
        print(f"Demo failed: {e}")


if __name__ == "__main__":
    demo_bridge()
