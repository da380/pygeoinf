"""
Radial Laplacian eigenfunction providers.

These providers generate eigenfunctions for the radial Laplacian operator:
    L = -d²/dr² - (2/r)d/dr

which is self-adjoint with respect to the weighted inner product:
    ⟨f,g⟩ = ∫ f(r)g(r) r² dr

The eigenfunctions depend on the domain and boundary conditions:
- Domain (0, R) with regularity at r=0: Dirichlet or Neumann at R
- Domain (a, b) with 0 < a < b: DD, DN, ND, or NN boundary conditions
"""

import numpy as np
import math
from .base import IndexedFunctionProvider
from ..utils.robin_utils import RobinRootFinder


class RadialLaplacianDirichletProvider(IndexedFunctionProvider):
    """
    Provider for radial Laplacian eigenfunctions on (0, R) with Dirichlet BC.

    Domain: (0, R) with regularity at r=0 and Dirichlet at r=R.
    Eigenfunctions: y_n(r) = φ_n(r)/r where φ_n(r) = √(2/R) sin(nπr/R)
    Eigenvalues: λ_n = (nπ/R)² for n=1,2,3,...
    """

    def __init__(self, space):
        """Initialize provider for (0,R) Dirichlet case."""
        super().__init__(space)
        self._cache = {}

        # Validate domain
        a = self.space.function_domain.a
        if not np.isclose(a, 0.0, atol=1e-10):
            raise ValueError(
                f"RadialLaplacianDirichletProvider requires domain starting at 0, "
                f"got a={a}"
            )

    def get_function_by_index(self, index: int):
        """
        Get eigenfunction at given index.

        Args:
            index: Function index (0-based, maps to n=1,2,3,...)

        Returns:
            Function: Radial eigenfunction y_n(r) = φ_n(r)/r
        """
        if index not in self._cache:
            R = self.space.function_domain.b
            n = index + 1  # n starts from 1

            # Normalization constant for φ_n(r) = c_n sin(nπr/R)
            c_n = np.sqrt(2.0 / R)

            def eigenfunction(r):
                """y_n(r) = c_n sin(nπr/R) / r"""
                r_arr = np.asarray(r)
                scalar_input = r_arr.ndim == 0
                r_arr = np.atleast_1d(r_arr)

                result = np.zeros_like(r_arr, dtype=float)
                nonzero = r_arr > 1e-14

                # For r > 0: y_n(r) = c_n sin(nπr/R) / r
                result[nonzero] = c_n * np.sin(n * np.pi * r_arr[nonzero] / R) / r_arr[nonzero]

                return result.item() if scalar_input else result

            from ..functions import Function
            func = Function(
                self.space,
                evaluate_callable=eigenfunction,
                name=f"y_{n}(r) [radial Dirichlet]"
            )
            self._cache[index] = func

        return self._cache[index]


class RadialLaplacianNeumannProvider(IndexedFunctionProvider):
    """
    Provider for radial Laplacian eigenfunctions on (0, R) with Neumann BC.

    Domain: (0, R) with regularity at r=0 and Neumann at r=R.

    Zero mode (index=0):
        y_0(r) = √(3/R³) (constant)
        λ_0 = 0

    Nonzero modes (index≥1):
        y_n(r) = c_n sin(k_n r) / r
        where k_n satisfies tan(k_n R) = k_n R
        Normalization: c_n = [R/2 - sin(2k_n R)/(4k_n)]^(-1/2)
        λ_n = k_n²
    """

    def __init__(self, space):
        """Initialize provider for (0,R) Neumann case."""
        super().__init__(space)
        self._cache = {}
        self._eigenvalue_cache = {}

        # Validate domain
        a = self.space.function_domain.a
        if not np.isclose(a, 0.0, atol=1e-10):
            raise ValueError(
                f"RadialLaplacianNeumannProvider requires domain starting at 0, "
                f"got a={a}"
            )

    def get_eigenvalue(self, index: int) -> float:
        """Get eigenvalue at given index."""
        if index not in self._eigenvalue_cache:
            R = self.space.function_domain.b

            if index == 0:
                eigenval = 0.0
            else:
                # Solve tan(kR) = kR for the index-th root
                F = lambda k: k * R
                k_root = RobinRootFinder.solve_tan_equation(F, R, index - 1)
                eigenval = k_root ** 2

            self._eigenvalue_cache[index] = eigenval

        return self._eigenvalue_cache[index]

    def get_function_by_index(self, index: int):
        """
        Get eigenfunction at given index.

        Args:
            index: Function index (0-based)
                   index=0 gives zero mode
                   index≥1 gives nonzero modes

        Returns:
            Function: Radial eigenfunction
        """
        if index not in self._cache:
            R = self.space.function_domain.b

            if index == 0:
                # Zero mode: y_0(r) = constant
                c_0 = np.sqrt(3.0 / (R ** 3))

                def eigenfunction(r):
                    """Zero mode: constant function"""
                    r_arr = np.asarray(r)
                    scalar_input = r_arr.ndim == 0
                    result = np.full_like(np.atleast_1d(r_arr), c_0, dtype=float)
                    return result.item() if scalar_input else result

                name = f"y_0(r) [radial Neumann zero mode]"
            else:
                # Nonzero mode: solve for k_n
                F = lambda k: k * R
                k_n = RobinRootFinder.solve_tan_equation(F, R, index - 1)

                # Normalization integral: I_n = R/2 - sin(2k_n R)/(4k_n)
                I_n = R / 2.0 - np.sin(2 * k_n * R) / (4 * k_n)
                c_n = 1.0 / np.sqrt(I_n)

                def eigenfunction(r):
                    """y_n(r) = c_n sin(k_n r) / r"""
                    r_arr = np.asarray(r)
                    scalar_input = r_arr.ndim == 0
                    r_arr = np.atleast_1d(r_arr)

                    result = np.zeros_like(r_arr, dtype=float)
                    nonzero = r_arr > 1e-14

                    result[nonzero] = c_n * np.sin(k_n * r_arr[nonzero]) / r_arr[nonzero]

                    return result.item() if scalar_input else result

                name = f"y_{index}(r) [radial Neumann]"

            from ..functions import Function
            func = Function(
                self.space,
                evaluate_callable=eigenfunction,
                name=name
            )
            self._cache[index] = func

        return self._cache[index]


class RadialLaplacianDDProvider(IndexedFunctionProvider):
    """
    Provider for radial Laplacian eigenfunctions on (a, b) with Dirichlet-Dirichlet BC.

    Domain: (a, b) with 0 < a < b, Dirichlet at both endpoints.
    Eigenfunctions: y_n(r) = √(2/L) sin(nπ(r-a)/L) / r
    Eigenvalues: λ_n = (nπ/L)² for n=1,2,3,...
    where L = b - a
    """

    def __init__(self, space):
        """Initialize provider for (a,b) Dirichlet-Dirichlet case."""
        super().__init__(space)
        self._cache = {}

        # Validate domain
        a = self.space.function_domain.a
        if not (a > 0):
            raise ValueError(
                f"RadialLaplacianDDProvider requires domain with a > 0, got a={a}"
            )

    def get_function_by_index(self, index: int):
        """
        Get eigenfunction at given index.

        Args:
            index: Function index (0-based, maps to n=1,2,3,...)

        Returns:
            Function: Radial eigenfunction y_n(r)
        """
        if index not in self._cache:
            a = self.space.function_domain.a
            b = self.space.function_domain.b
            L = b - a
            n = index + 1  # n starts from 1

            c_n = np.sqrt(2.0 / L)

            def eigenfunction(r):
                """y_n(r) = √(2/L) sin(nπ(r-a)/L) / r"""
                r_arr = np.asarray(r)
                scalar_input = r_arr.ndim == 0
                r_arr = np.atleast_1d(r_arr)

                result = c_n * np.sin(n * np.pi * (r_arr - a) / L) / r_arr

                return result.item() if scalar_input else result

            from ..functions import Function
            func = Function(
                self.space,
                evaluate_callable=eigenfunction,
                name=f"y_{n}(r) [radial DD]"
            )
            self._cache[index] = func

        return self._cache[index]


class RadialLaplacianDNProvider(IndexedFunctionProvider):
    """
    Provider for radial Laplacian eigenfunctions on (a, b) with Dirichlet-Neumann BC.

    Domain: (a, b) with 0 < a < b, Dirichlet at a, Neumann at b.
    Eigenfunctions: y_n(r) = c_n sin(k_n(r-a)) / r
    where k_n satisfies tan(k_n L) = k_n b
    Normalization: c_n = [L/2 - sin(2k_n L)/(4k_n)]^(-1/2)
    """

    def __init__(self, space):
        """Initialize provider for (a,b) Dirichlet-Neumann case."""
        super().__init__(space)
        self._cache = {}
        self._eigenvalue_cache = {}

        # Validate domain
        a = self.space.function_domain.a
        if not (a > 0):
            raise ValueError(
                f"RadialLaplacianDNProvider requires domain with a > 0, got a={a}"
            )

    def get_eigenvalue(self, index: int) -> float:
        """Get eigenvalue at given index."""
        if index not in self._eigenvalue_cache:
            a = self.space.function_domain.a
            b = self.space.function_domain.b
            L = b - a

            # Solve tan(kL) = kb
            F = lambda k: k * b
            k_root = RobinRootFinder.solve_tan_equation(F, L, index)
            eigenval = k_root ** 2

            self._eigenvalue_cache[index] = eigenval

        return self._eigenvalue_cache[index]

    def get_function_by_index(self, index: int):
        """
        Get eigenfunction at given index.

        Args:
            index: Function index (0-based)

        Returns:
            Function: Radial eigenfunction y_n(r)
        """
        if index not in self._cache:
            a = self.space.function_domain.a
            b = self.space.function_domain.b
            L = b - a

            # Solve tan(kL) = kb
            F = lambda k: k * b
            k_n = RobinRootFinder.solve_tan_equation(F, L, index)

            # Normalization: I_n = L/2 - sin(2k_n L)/(4k_n)
            I_n = L / 2.0 - np.sin(2 * k_n * L) / (4 * k_n)
            c_n = 1.0 / np.sqrt(I_n)

            def eigenfunction(r):
                """y_n(r) = c_n sin(k_n(r-a)) / r"""
                r_arr = np.asarray(r)
                scalar_input = r_arr.ndim == 0
                r_arr = np.atleast_1d(r_arr)

                result = c_n * np.sin(k_n * (r_arr - a)) / r_arr

                return result.item() if scalar_input else result

            from ..functions import Function
            func = Function(
                self.space,
                evaluate_callable=eigenfunction,
                name=f"y_{index}(r) [radial DN]"
            )
            self._cache[index] = func

        return self._cache[index]


class RadialLaplacianNDProvider(IndexedFunctionProvider):
    """
    Provider for radial Laplacian eigenfunctions on (a, b) with Neumann-Dirichlet BC.

    Domain: (a, b) with 0 < a < b, Neumann at a, Dirichlet at b.
    Eigenfunctions: y_n(r) = c_n sin(k_n(b-r)) / r
    where k_n satisfies tan(k_n L) = -a k_n
    Normalization: c_n = [L/2 - sin(2k_n L)/(4k_n)]^(-1/2)
    """

    def __init__(self, space):
        """Initialize provider for (a,b) Neumann-Dirichlet case."""
        super().__init__(space)
        self._cache = {}
        self._eigenvalue_cache = {}

        # Validate domain
        a = self.space.function_domain.a
        if not (a > 0):
            raise ValueError(
                f"RadialLaplacianNDProvider requires domain with a > 0, got a={a}"
            )

    def get_eigenvalue(self, index: int) -> float:
        """Get eigenvalue at given index."""
        if index not in self._eigenvalue_cache:
            a = self.space.function_domain.a
            b = self.space.function_domain.b
            L = b - a

            # Solve tan(kL) = -ak
            F = lambda k: -a * k
            k_root = RobinRootFinder.solve_tan_equation(F, L, index)
            eigenval = k_root ** 2

            self._eigenvalue_cache[index] = eigenval

        return self._eigenvalue_cache[index]

    def get_function_by_index(self, index: int):
        """
        Get eigenfunction at given index.

        Args:
            index: Function index (0-based)

        Returns:
            Function: Radial eigenfunction y_n(r)
        """
        if index not in self._cache:
            a = self.space.function_domain.a
            b = self.space.function_domain.b
            L = b - a

            # Solve tan(kL) = -ak
            F = lambda k: -a * k
            k_n = RobinRootFinder.solve_tan_equation(F, L, index)

            # Normalization: I_n = L/2 - sin(2k_n L)/(4k_n)
            I_n = L / 2.0 - np.sin(2 * k_n * L) / (4 * k_n)
            c_n = 1.0 / np.sqrt(I_n)

            def eigenfunction(r):
                """y_n(r) = c_n sin(k_n(b-r)) / r"""
                r_arr = np.asarray(r)
                scalar_input = r_arr.ndim == 0
                r_arr = np.atleast_1d(r_arr)

                result = c_n * np.sin(k_n * (b - r_arr)) / r_arr

                return result.item() if scalar_input else result

            from ..functions import Function
            func = Function(
                self.space,
                evaluate_callable=eigenfunction,
                name=f"y_{index}(r) [radial ND]"
            )
            self._cache[index] = func

        return self._cache[index]


class RadialLaplacianNNProvider(IndexedFunctionProvider):
    """
    Provider for radial Laplacian eigenfunctions on (a, b) with Neumann-Neumann BC.

    Domain: (a, b) with 0 < a < b, Neumann at both endpoints.

    Zero mode (index=0):
        y_0(r) = √(3/(b³-a³)) (constant)
        λ_0 = 0

    Nonzero modes (index≥1):
        y_n(r) = c_n [sin(k_n(r-a)) + ak_n cos(k_n(r-a))] / r
        where k_n satisfies tan(k_n L) = (1/b - 1/a) / (k + 1/(abk))
        Normalization from integral calculation
    """

    def __init__(self, space):
        """Initialize provider for (a,b) Neumann-Neumann case."""
        super().__init__(space)
        self._cache = {}
        self._eigenvalue_cache = {}

        # Validate domain
        a = self.space.function_domain.a
        if not (a > 0):
            raise ValueError(
                f"RadialLaplacianNNProvider requires domain with a > 0, got a={a}"
            )

    def get_eigenvalue(self, index: int) -> float:
        """Get eigenvalue at given index."""
        if index not in self._eigenvalue_cache:
            if index == 0:
                eigenval = 0.0
            else:
                a = self.space.function_domain.a
                b = self.space.function_domain.b
                L = b - a

                # Solve tan(kL) = (1/b - 1/a) / (k + 1/(abk))
                numerator = 1.0/b - 1.0/a
                F = lambda k: numerator / (k + 1.0/(a * b * k))
                k_root = RobinRootFinder.solve_tan_equation(F, L, index - 1)
                eigenval = k_root ** 2

            self._eigenvalue_cache[index] = eigenval

        return self._eigenvalue_cache[index]

    def get_function_by_index(self, index: int):
        """
        Get eigenfunction at given index.

        Args:
            index: Function index (0-based)
                   index=0 gives zero mode
                   index≥1 gives nonzero modes

        Returns:
            Function: Radial eigenfunction y_n(r)
        """
        if index not in self._cache:
            a = self.space.function_domain.a
            b = self.space.function_domain.b
            L = b - a

            if index == 0:
                # Zero mode: y_0(r) = constant
                c_0 = np.sqrt(3.0 / (b**3 - a**3))

                def eigenfunction(r):
                    """Zero mode: constant function"""
                    r_arr = np.asarray(r)
                    scalar_input = r_arr.ndim == 0
                    result = np.full_like(np.atleast_1d(r_arr), c_0, dtype=float)
                    return result.item() if scalar_input else result

                name = f"y_0(r) [radial NN zero mode]"
            else:
                # Nonzero mode: solve for k_n
                numerator = 1.0/b - 1.0/a
                F = lambda k: numerator / (k + 1.0/(a * b * k))
                k_n = RobinRootFinder.solve_tan_equation(F, L, index - 1)

                # Normalization: I_n = L/2*(1+(ak)²) + sin(2kL)/(4k)*((ak)²-1) + ak*(1-cos(2kL))/(2k)
                ak = a * k_n
                I_n = (L / 2.0 * (1 + ak**2) +
                       np.sin(2 * k_n * L) / (4 * k_n) * (ak**2 - 1) +
                       ak * (1 - np.cos(2 * k_n * L)) / (2 * k_n))
                c_n = 1.0 / np.sqrt(I_n)

                def eigenfunction(r):
                    """y_n(r) = c_n [sin(k_n(r-a)) + ak_n cos(k_n(r-a))] / r"""
                    r_arr = np.asarray(r)
                    scalar_input = r_arr.ndim == 0
                    r_arr = np.atleast_1d(r_arr)

                    u_k = np.sin(k_n * (r_arr - a)) + ak * np.cos(k_n * (r_arr - a))
                    result = c_n * u_k / r_arr

                    return result.item() if scalar_input else result

                name = f"y_{index}(r) [radial NN]"

            from ..functions import Function
            func = Function(
                self.space,
                evaluate_callable=eigenfunction,
                name=name
            )
            self._cache[index] = func

        return self._cache[index]
