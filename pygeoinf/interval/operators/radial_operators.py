"""
Radial Laplacian operators for 3D spherical coordinates.

These operators are self-adjoint with respect to the weighted inner product
⟨f,g⟩ = ∫ f(r)g(r) r² dr, which arises naturally in spherical coordinates.
"""

import logging
import numpy as np
from typing import Union, Optional, Literal
from scipy.sparse import diags

from . import SpectralOperator
from ..lebesgue_space import Lebesgue
from ..sobolev_space import Sobolev
from ..boundary_conditions import BoundaryConditions
from ..functions import Function
from ..providers import SpectrumProvider, EigenvalueProvider
from ..configs import IntegrationConfig
from ..utils.robin_utils import RobinRootFinder


class RadialLaplacianEigenvalueProvider(EigenvalueProvider):
    """
    Eigenvalue provider for the radial Laplacian operator.

    The radial Laplacian in 3D spherical coordinates is:
        L = -d²/dr² - (2/r)d/dr = -(1/r²)d/dr(r² d/dr)

    For the weighted inner product ⟨f,g⟩ = ∫ f(r)g(r) r² dr,
    this operator is self-adjoint.

    Eigenvalues and eigenfunctions depend on boundary conditions:
    - Dirichlet at r=0 and r=R: Related to spherical Bessel functions
    - Neumann at r=R (with regularity at r=0): Different eigenstructure
    """

    def __init__(
        self,
        function_domain,
        boundary_conditions: BoundaryConditions,
        inverse: bool = False,
        alpha: float = 1.0,
        ell: int = 0
    ):
        """
        Initialize the radial Laplacian eigenvalue provider.

        Args:
            function_domain: Interval domain [0, R]
            boundary_conditions: Boundary conditions at r=R
            inverse: If True, compute eigenvalues of L⁻¹, else of L
            alpha: Scaling factor for the eigenvalues (default: 1.0)
            ell: Angular momentum quantum number (default: 0, s-wave)
        """
        self._function_domain = function_domain
        self._boundary_conditions = boundary_conditions
        self._inverse = inverse
        self._alpha = alpha
        self._ell = ell  # Angular momentum number
        self._eigenvalue_cache = {}
        self._eigenvalue_numerical = None  # Cache for numerical eigenvalues

    def get_eigenvalue(self, index: int) -> float:
        """Get eigenvalue for given index."""
        if index not in self._eigenvalue_cache:
            self._eigenvalue_cache[index] = self._compute_eigenvalue(index)
        return self._eigenvalue_cache[index]

    def _compute_eigenvalue(self, index: int) -> float:
        """
        Compute eigenvalue based on boundary conditions.

        For ℓ=0 (s-wave):

        Case A: Domain (0, R) with regularity at r=0:
            1. regularity-Dirichlet: λₖ = (kπ/R)² for k=1,2,3,...
            2. regularity-Neumann: solve tan(kR) = kR numerically

        Case B: Domain (a, b) with 0 < a < b:
            The wavenumbers k=√λ satisfy:
            1. DD: k_n = nπ/L for n=1,2,3,... (analytical)
            2. DN: tan(kL) = kb (numerical)
            3. ND: tan(kL) = -ak (numerical)
            4. NN: tan(kL) = (1/b - 1/a)/(k + 1/(abk)), λ_0=0 (numerical)

            where L = b - a.
            Eigenfunctions: u_k(r) = A·sin(k(r-a)) + B·cos(k(r-a))
            Scaled eigenfunctions: y_k(r) = u_k(r)/r
            Orthogonal in L²((a,b); r²dr)

        For general ℓ:
            Eigenvalues from zeros of spherical Bessel functions
            (not yet implemented)
        """
        a = self._function_domain.a
        b = self._function_domain.b
        L = b - a

        if self._ell == 0:
            # s-wave (ℓ=0): analytical or semi-analytical eigenvalues

            # Case A: Domain includes r=0 (regularity condition at left)
            if np.isclose(a, 0.0, atol=1e-10):
                if self._boundary_conditions.type == 'dirichlet':
                    # regularity-Dirichlet: φ(r) ∝ sin(kr)/r, φ(R)=0
                    # Eigenvalues: (kπ/R)² for k=1,2,3,...
                    k = index + 1
                    eigenval = (k * np.pi / L) ** 2

                elif self._boundary_conditions.type == 'neumann':
                    # regularity-Neumann: φ(r) ∝ sin(kr)/r, φ'(R)=0
                    # d/dr[sin(kr)/r] = (kr*cos(kr) - sin(kr))/r²
                    # At r=R: kR*cos(kR) - sin(kR) = 0 → tan(kR) = kR
                    eigenval = self._compute_regularity_neumann_eigenvalue(
                        index, L)

                else:
                    raise ValueError(
                        f"For domain containing r=0, only 'dirichlet' "
                        f"(regularity-Dirichlet) or 'neumann' "
                        f"(regularity-Neumann) are supported. "
                        f"Got: '{self._boundary_conditions.type}'"
                    )

            # Case B: Domain does not include r=0 (two-endpoint BCs)
            else:
                if self._boundary_conditions.type == 'dirichlet':
                    # DD: u(a)=0, u(b)=0
                    # k_n = nπ/L for n=1,2,3,...
                    k = (index + 1) * np.pi / L
                    eigenval = k ** 2

                elif self._boundary_conditions.type == 'mixed_dirichlet_neumann':
                    # DN: u(a)=0, u'(b)=0
                    # tan(kL) = kb
                    eigenval = self._compute_dn_eigenvalue(index, a, b, L)

                elif self._boundary_conditions.type == 'mixed_neumann_dirichlet':
                    # ND: u'(a)=0, u(b)=0
                    # tan(kL) = -ak
                    eigenval = self._compute_nd_eigenvalue(index, a, b, L)

                elif self._boundary_conditions.type == 'neumann':
                    # NN: u'(a)=0, u'(b)=0
                    # tan(kL) = (1/b - 1/a) / (k + 1/(abk))
                    # λ_0 = 0 with constant eigenfunction
                    if index == 0:
                        eigenval = 0.0
                    else:
                        eigenval = self._compute_nn_eigenvalue(
                            index, a, b, L)

                elif self._boundary_conditions.type == 'robin':
                    # Robin BC: not yet implemented for radial case
                    raise NotImplementedError(
                        "Robin boundary conditions not yet implemented "
                        "for radial Laplacian"
                    )

                else:
                    raise ValueError(
                        f"Unsupported boundary condition type "
                        f"'{self._boundary_conditions.type}' "
                        f"for radial Laplacian on domain not containing r=0"
                    )
        else:
            # General ℓ: use spherical Bessel functions
            raise NotImplementedError(
                f"Radial Laplacian for ℓ={self._ell} not yet implemented. "
                f"Currently only ℓ=0 (s-wave) is supported."
            )

        # Apply alpha scaling and inverse if needed
        if self._inverse:
            if eigenval == 0:
                raise ValueError("Cannot invert zero eigenvalue")
            return 1.0 / (eigenval * self._alpha)
        else:
            return eigenval * self._alpha

    def _compute_regularity_neumann_eigenvalue(
            self, index: int, R: float
    ) -> float:
        """
        Compute regularity-Neumann eigenvalue for ℓ=0.

        Solve tan(kR) = kR for the (index+1)-th root.
        This is for domain (0, R) with regularity at r=0
        and Neumann at r=R.

        Args:
            index: Eigenvalue index (0-based)
            R: Domain radius

        Returns:
            Eigenvalue λ = k²
        """
        # tan(kR) = kR
        if index == 0:
            return 0.0  # First root is k=0 → λ=0
        else:
            F = lambda k: k * R
            k_root = RobinRootFinder.solve_tan_equation(F, R, index - 1)
            return k_root ** 2



    def _compute_dn_eigenvalue(
            self, index: int, a: float, b: float, L: float
    ) -> float:
        """
        Compute Dirichlet-Neumann eigenvalue for ℓ=0 on (a,b).

        Solve tan(kL) = kb for the (index+1)-th root.

        Args:
            index: Eigenvalue index (0-based)
            a: Left endpoint
            b: Right endpoint
            L: Domain length (b-a)

        Returns:
            Eigenvalue λ = k²
        """
        # tan(kL) = kb
        F = lambda k: k * b
        k_root = RobinRootFinder.solve_tan_equation(F, L, index-1)
        return k_root ** 2

    def _compute_nd_eigenvalue(
            self, index: int, a: float, b: float, L: float
    ) -> float:
        """
        Compute Neumann-Dirichlet eigenvalue for ℓ=0 on (a,b).

        Solve tan(kL) = -ak for the (index+1)-th root.

        Args:
            index: Eigenvalue index (0-based)
            a: Left endpoint
            b: Right endpoint
            L: Domain length (b-a)

        Returns:
            Eigenvalue λ = k²
        """
        # tan(kL) = -ak
        F = lambda k: -a * k
        k_root = RobinRootFinder.solve_tan_equation(F, L, index)
        return k_root ** 2

    def _compute_nn_eigenvalue(
            self, index: int, a: float, b: float, L: float
    ) -> float:
        """
        Compute Neumann-Neumann eigenvalue for ℓ=0 on (a,b).

        Solve tan(kL) = (1/b - 1/a) / (k + 1/(abk)) for the index-th root.
        Note: λ_0 = 0 is handled separately in caller.

        Args:
            index: Eigenvalue index (1-based for non-zero eigenvalues)
            a: Left endpoint
            b: Right endpoint
            L: Domain length (b-a)

        Returns:
            Eigenvalue λ = k²
        """
        # tan(kL) = (1/b - 1/a) / (k + 1/(abk))
        numerator = 1.0/b - 1.0/a
        F = lambda k: numerator / (k + 1.0/(a * b * k))
        # index is 1-based for nonzero modes (index=1 is first nonzero)
        # so solve for (index-1)-th root
        k_root = RobinRootFinder.solve_tan_equation(F, L, index - 1)
        return k_root ** 2


class RadialLaplacianSpectrumProvider(SpectrumProvider):
    """
    Spectrum provider for radial Laplacian eigenfunctions and eigenvalues.

    This provider delegates to the appropriate function provider based on
    the domain and boundary conditions.
    """

    def __init__(
        self,
        space: Union[Lebesgue, Sobolev],
        boundary_conditions: BoundaryConditions,
        alpha: float = 1.0,
        inverse: bool = False,
        ell: int = 0
    ):
        """
        Initialize radial Laplacian spectrum provider.

        Args:
            space: Function space (Lebesgue or Sobolev)
            boundary_conditions: Boundary conditions
            alpha: Scaling factor
            inverse: If True, spectrum of inverse operator
            ell: Angular momentum quantum number
        """
        self._boundary_conditions = boundary_conditions
        self._inverse = inverse
        self._ell = ell
        super().__init__(space, orthonormal=True, basis_type='radial_laplacian')

        self._eigenvalue_provider = RadialLaplacianEigenvalueProvider(
            space.function_domain,
            boundary_conditions,
            inverse,
            alpha,
            ell
        )

        # Initialize the appropriate function provider
        self._function_provider = self._create_function_provider()

    def _create_function_provider(self):
        """Create the appropriate function provider based on domain and BC."""
        from ..function_providers import (
            RadialLaplacianDirichletProvider,
            RadialLaplacianNeumannProvider,
            RadialLaplacianDDProvider,
            RadialLaplacianDNProvider,
            RadialLaplacianNDProvider,
            RadialLaplacianNNProvider,
        )

        a = self.space.function_domain.a
        bc_type = self._boundary_conditions.type

        # Case A: Domain (0, R) with regularity at r=0
        if np.isclose(a, 0.0, atol=1e-10):
            if bc_type == 'dirichlet':
                return RadialLaplacianDirichletProvider(self.space)
            elif bc_type == 'neumann':
                return RadialLaplacianNeumannProvider(self.space)
            else:
                raise NotImplementedError(
                    f"Radial Laplacian on (0,R) with BC type '{bc_type}' not implemented"
                )

        # Case B: Domain (a, b) with 0 < a < b
        else:
            if bc_type == 'dirichlet':
                return RadialLaplacianDDProvider(self.space)
            elif bc_type == 'mixed_dirichlet_neumann':
                return RadialLaplacianDNProvider(self.space)
            elif bc_type == 'mixed_neumann_dirichlet':
                return RadialLaplacianNDProvider(self.space)
            elif bc_type == 'neumann':
                return RadialLaplacianNNProvider(self.space)
            else:
                raise NotImplementedError(
                    f"Radial Laplacian on (a,b) with BC type '{bc_type}' not implemented"
                )

    def get_eigenvalue(self, index: int) -> float:
        """Get eigenvalue at given index."""
        return self._eigenvalue_provider.get_eigenvalue(index)

    def get_eigenfunction(self, index: int) -> Function:
        """
        Get eigenfunction at given index.

        Delegates to the appropriate function provider based on domain and BC.
        These are orthonormal with respect to ⟨f,g⟩ = ∫ f(r)g(r) r² dr.
        """
        return self._function_provider.get_function_by_index(index)


class RadialLaplacian(SpectralOperator):
    """
    Radial Laplacian operator for 3D spherical coordinates.

    The radial Laplacian is:
        L = -d²/dr² - (2/r)d/dr = -(1/r²)d/dr(r² d/dr)

    This operator is self-adjoint with respect to the weighted inner product:
        ⟨f,g⟩ = ∫ f(r)g(r) r² dr

    The weight r² comes from the Jacobian in spherical coordinates.

    Multiple discretization methods are available:
    - 'spectral': Uses analytical eigendecomposition (ℓ=0 only for now)
    - 'fd': Finite difference on a radial grid
    """

    def __init__(
        self,
        domain: Union[Lebesgue, Sobolev],
        boundary_conditions: BoundaryConditions,
        alpha: float = 1.0,
        /,
        *,
        method: Literal['spectral', 'fd'] = 'spectral',
        dofs: Optional[int] = None,
        ell: int = 0,
        fd_order: int = 2,
        n_samples: int = 512,
        integration_config: IntegrationConfig,
    ):
        """
        Initialize the radial Laplacian operator.

        Args:
            domain: Function space (must have weight=r²)
            boundary_conditions: Boundary conditions at outer radius
            alpha: Scaling factor (default: 1.0)
            method: Discretization method ('spectral' or 'fd')
            dofs: Number of degrees of freedom
            ell: Angular momentum quantum number (default: 0)
            fd_order: Order of finite difference stencil (2, 4, 6)
            n_samples: Number of samples for spectral transforms
            integration_config: Integration configuration
        """
        self._domain = domain
        self._boundary_conditions = boundary_conditions
        self._alpha = alpha
        self._dofs = dofs if dofs is not None else domain.dim
        self._ell = ell
        self._fd_order = fd_order
        self._method = method
        self._n_samples = max(n_samples, self._dofs)

        # Store integration config
        self.integration = integration_config

        super().__init__(domain, domain, self._apply)

        # Initialize spectrum provider for spectral method
        if method == 'spectral':
            self._spectrum_provider = RadialLaplacianSpectrumProvider(
                domain,
                boundary_conditions,
                alpha,
                inverse=False,
                ell=ell
            )
        elif method == 'fd':
            self._setup_finite_difference()
        else:
            raise ValueError(f"Unknown method: {method}")

        # Logger
        self._log = logging.getLogger(__name__)
        self._log.info(
            "RadialLaplacian initialized: method=%s, dofs=%s, ℓ=%s, alpha=%s",
            method, self._dofs, ell, alpha
        )

    def get_eigenvalue(self, index: int) -> float:
        """Get the eigenvalue at a specific index."""
        if self._method == 'spectral':
            return self._spectrum_provider.get_eigenvalue(index)
        else:
            raise NotImplementedError(
                "Eigenvalues not available for finite difference method"
            )

    def get_eigenfunction(self, index: int) -> Function:
        """Get the eigenfunction at a specific index."""
        if self._method == 'spectral':
            return self._spectrum_provider.get_eigenfunction(index)
        else:
            raise NotImplementedError(
                "Eigenfunctions not available for finite difference method"
            )

    def _setup_finite_difference(self):
        """Setup finite difference discretization for radial Laplacian."""
        a, b = self._domain.function_domain.a, self._domain.function_domain.b

        # Create radial grid
        # Need to handle r=0 carefully - either start slightly away or use special stencil
        if a == 0:
            # Start grid slightly away from zero to avoid singularity
            eps = (b - a) / (10 * self._dofs)
            self._r_grid = np.linspace(eps, b, self._dofs)
        else:
            self._r_grid = np.linspace(a, b, self._dofs)

        self._dr = self._r_grid[1] - self._r_grid[0]

        # Create FD matrix for radial Laplacian
        self._fd_matrix = self._create_radial_fd_matrix()

        self._log.info(
            "RadialLaplacian FD setup: grid from r=%.3e to r=%.3e, dr=%.3e",
            self._r_grid[0], self._r_grid[-1], self._dr
        )

    def _create_radial_fd_matrix(self):
        """
        Create finite difference matrix for the radial Laplacian.

        L = -d²/dr² - (2/r)d/dr

        Using centered differences:
            d²f/dr² ≈ (f_{i+1} - 2f_i + f_{i-1})/dr²
            df/dr ≈ (f_{i+1} - f_{i-1})/(2dr)

        So: L_i f ≈ -[f_{i+1} - 2f_i + f_{i-1}]/dr² - (2/r_i)[f_{i+1} - f_{i-1}]/(2dr)
                   = -[(1 + dr/r_i)f_{i+1} - 2f_i + (1 - dr/r_i)f_{i-1}]/dr²
        """
        n = self._dofs
        dr = self._dr
        r = self._r_grid

        # Main diagonal
        main_diag = np.full(n, 2.0 / dr**2)

        # Upper diagonal: -(1 + dr/r_i)/dr²
        upper_diag = -(1.0 + dr / r[:-1]) / dr**2

        # Lower diagonal: -(1 - dr/r_i)/dr²
        lower_diag = -(1.0 - dr / r[1:]) / dr**2

        # Create sparse matrix
        matrix = diags(
            [lower_diag, main_diag, upper_diag],
            offsets=[-1, 0, 1],
            shape=(n, n),
            format='csr'
        )

        # Apply boundary conditions
        if self._boundary_conditions.type == 'dirichlet':
            # f(R) = 0
            # Already handled by the stencil if we don't modify last row
            pass
        elif self._boundary_conditions.type in ['neumann', 'mixed_neumann_dirichlet']:
            # df/dr(R) = 0
            # Use one-sided stencil at boundary
            # f'(R) ≈ (-3f_n + 4f_{n-1} - f_{n-2})/(2dr) = 0
            # This gives: f_n = (4f_{n-1} - f_{n-2})/3
            # Can be incorporated into the matrix or handled separately
            pass

        return matrix.toarray()

    def _apply(self, f: Function) -> Function:
        """Apply the radial Laplacian operator to a function."""
        if self._method == 'spectral':
            return self._apply_spectral(f)
        elif self._method == 'fd':
            return self._apply_fd(f)
        else:
            raise ValueError(f"Unknown method: {self._method}")

    def _apply_spectral(self, f: Function) -> Function:
        """Apply radial Laplacian using spectral decomposition."""
        # Expand f in eigenbasis
        coeffs = []
        for k in range(self._dofs):
            phi_k = self.get_eigenfunction(k)
            # Project: c_k = ⟨φ_k, f⟩ with weighted inner product
            c_k = self._domain.inner_product(phi_k, f)
            coeffs.append(c_k)

        # Apply operator: L f = Σ λ_k c_k φ_k
        result = self._domain.zero
        for k, c_k in enumerate(coeffs):
            if abs(c_k) > 1e-14:  # Skip near-zero coefficients
                lambda_k = self.get_eigenvalue(k)
                phi_k = self.get_eigenfunction(k)
                result = result + (lambda_k * c_k) * phi_k

        return result

    def _apply_fd(self, f: Function) -> Function:
        """Apply radial Laplacian using finite differences."""
        # Evaluate function on grid
        f_values = f.evaluate(self._r_grid)

        # Apply FD matrix
        laplacian_values = self._fd_matrix @ f_values

        # Create result function by interpolation
        def laplacian_func(r):
            return np.interp(r, self._r_grid, laplacian_values)

        return Function(self.codomain, evaluate_callable=laplacian_func)


class InverseRadialLaplacian(SpectralOperator):
    """
    Inverse radial Laplacian operator for use as prior covariance.

    This operator solves:
        L u = f
    where L is the radial Laplacian.

    It provides a self-adjoint, positive-definite operator suitable
    for defining Gaussian measures on spaces with r² weight.
    """

    def __init__(
        self,
        domain: Union[Lebesgue, Sobolev],
        boundary_conditions: BoundaryConditions,
        alpha: float = 1.0,
        /,
        *,
        method: Literal['fem', 'spectral'] = 'spectral',
        dofs: int = 100,
        ell: int = 0,
        fem_type: str = "hat",
        n_samples: int = 512,
        integration_config: IntegrationConfig,
    ):
        """
        Initialize the inverse radial Laplacian operator.

        Args:
            domain: Function space (must have weight=r²)
            boundary_conditions: Boundary conditions
            alpha: Scaling factor (default: 1.0)
            method: Solution method ('fem' or 'spectral')
            dofs: Number of degrees of freedom
            ell: Angular momentum quantum number (default: 0)
            fem_type: FEM type ('hat' or 'general')
            n_samples: Number of samples for spectral transforms
            integration_config: Integration configuration
        """
        if not isinstance(domain, (Lebesgue, Sobolev)):
            raise TypeError(
                f"domain must be a Lebesgue or Sobolev space, got {type(domain)}"
            )

        self._domain = domain
        self._boundary_conditions = boundary_conditions
        self._alpha = alpha
        self._method = method
        self._dofs = dofs if dofs is not None else domain.dim
        self._ell = ell
        self._fem_type = fem_type
        self._n_samples = max(n_samples, self._dofs)

        # Store integration config
        self.integration = integration_config

        super().__init__(domain, domain, self._apply)

        # Initialize spectrum provider for spectral method
        if method == 'spectral':
            self._spectrum_provider = RadialLaplacianSpectrumProvider(
                domain,
                boundary_conditions,
                alpha,
                inverse=True,
                ell=ell
            )
        elif method == 'fem':
            self._initialize_fem_solver()
        else:
            raise ValueError(f"Unknown method: {method}")

        # Logger
        self._log = logging.getLogger(__name__)
        self._log.info(
            "InverseRadialLaplacian initialized: method=%s, dofs=%s, ℓ=%s, alpha=%s",
            method, dofs, ell, alpha
        )

    def _initialize_fem_solver(self):
        """Initialize FEM solver for the inverse radial Laplacian."""
        # This would require a weighted FEM solver
        # For now, raise NotImplementedError
        raise NotImplementedError(
            "FEM solver for inverse radial Laplacian not yet implemented. "
            "Use method='spectral' instead."
        )

    def get_eigenvalue(self, index: int) -> float:
        """Get eigenvalue of the inverse operator."""
        if self._method == 'spectral':
            return self._spectrum_provider.get_eigenvalue(index)
        else:
            raise NotImplementedError(
                "Eigenvalues not available for FEM method"
            )

    def get_eigenfunction(self, index: int) -> Function:
        """Get eigenfunction (same as forward operator)."""
        if self._method == 'spectral':
            return self._spectrum_provider.get_eigenfunction(index)
        else:
            raise NotImplementedError(
                "Eigenfunctions not available for FEM method"
            )

    def get_eigenvalues(self, indices) -> np.ndarray:
        """Get multiple eigenvalues."""
        return np.array([self.get_eigenvalue(i) for i in indices])

    def _apply(self, f: Function) -> Function:
        """Apply the inverse radial Laplacian to a function."""
        if self._method == 'spectral':
            return self._apply_spectral(f)
        elif self._method == 'fem':
            return self._apply_fem(f)
        else:
            raise ValueError(f"Unknown method: {self._method}")

    def _apply_spectral(self, f: Function) -> Function:
        """Apply inverse using spectral decomposition."""
        # Expand f in eigenbasis
        coeffs = []
        for k in range(self._dofs):
            phi_k = self.get_eigenfunction(k)
            # Project: c_k = ⟨φ_k, f⟩ with weighted inner product
            c_k = self._domain.inner_product(phi_k, f)
            coeffs.append(c_k)

        # Apply inverse operator: L⁻¹ f = Σ (1/λ_k) c_k φ_k
        result = self._domain.zero
        for k, c_k in enumerate(coeffs):
            if abs(c_k) > 1e-14:
                inv_lambda_k = self.get_eigenvalue(k)  # Already inverted
                phi_k = self.get_eigenfunction(k)
                result = result + (inv_lambda_k * c_k) * phi_k

        return result

    def _apply_fem(self, f: Function) -> Function:
        """Apply inverse using FEM solver."""
        raise NotImplementedError(
            "FEM solver for inverse radial Laplacian not yet implemented"
        )
