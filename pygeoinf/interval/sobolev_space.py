"""
Sobolev spaces on a segment/interval [a, b].
"""

from typing import TYPE_CHECKING, Optional, Union, List
from pygeoinf import MassWeightedHilbertSpace, HilbertSpaceDirectSum
from .functions import Function
from .linear_form_kernel import LinearFormKernel

import numpy as np

# Import types for annotations but avoid runtime circular import
if TYPE_CHECKING:
    from pygeoinf.interval import Lebesgue, SpectralOperator, IntervalDomain


class Sobolev(MassWeightedHilbertSpace):
    def __init__(
        self,
        dim: int,
        function_domain: "IntervalDomain",
        s: float,
        k: float,
        L: "SpectralOperator",
        /, *,
        basis: Optional[Union[str, list]] = None,
        integration_config=None,
        parallel_config=None,
    ):
        """
        Initialize a Sobolev space.

        Args:
            dim: Dimension of the space
            function_domain: Domain for functions
            s: Regularity parameter (order)
            k: Scaling parameter
            L: Spectral operator (typically Laplacian)
            basis: Basis type for underlying Lebesgue space
            integration_config: Integration configuration (passed to
                underlying Lebesgue space). Can be IntegrationConfig or
                LebesgueIntegrationConfig. If None, extracts from L if
                available.
            parallel_config: Parallel configuration (passed to underlying
                Lebesgue space). Can be ParallelConfig or
                LebesgueParallelConfig.
        """
        # Extract configs from Laplacian operator if not provided
        if integration_config is None and hasattr(L, '_integration_config'):
            integration_config = L._integration_config
        if parallel_config is None and hasattr(L, '_parallel_config'):
            parallel_config = L._parallel_config

        # Attributes if Sobolev space
        self._underlying_space = self._create_underlying_space(
            dim, function_domain, basis, integration_config, parallel_config
        )
        self._s = s
        self._k = k
        self._L = L

        self._dofs = L._dofs

        # Atributes borrowed from Lebesgue
        self._function_domain = self._underlying_space.function_domain

        # Import at runtime to avoid circular import
        from pygeoinf.interval.operators import (
            BesselSobolev, BesselSobolevInverse
        )

        # Create the mass operator M = (k^2 I + A)^s
        M_op = BesselSobolev(
            self._underlying_space,
            self._underlying_space,
            k=self._k,
            s=2 * self._s,
            L=L,
            dofs=self._dofs
        )
        M_op_inv = BesselSobolevInverse(
            self._underlying_space,
            self._underlying_space,
            k=self._k,
            s=2 * self._s,
            L=L,
            dofs=self._dofs
        )
        # Compute the inverse mass operator
        super().__init__(
            self._underlying_space,
            M_op,
            M_op_inv
        )

    @property
    def function_domain(self):
        return self._function_domain

    @property
    def zero(self):
        return Function(self, evaluate_callable=lambda x: np.zeros_like(x))

    def to_dual(self, x: 'Function') -> 'LinearFormKernel':
        if not isinstance(x, Function):
            raise TypeError("Expected Function for primal element")
        kernel = self._mass_operator(x)
        # Get integration config from underlying Lebesgue space
        int_cfg = self._underlying_space.integration.dual
        par_cfg = self._underlying_space.parallel.dual
        return LinearFormKernel(
            self,
            kernel=kernel,
            integration_config=int_cfg,
            parallel_config=par_cfg
        )

    def from_dual(self, xp: 'LinearFormKernel') -> 'Function':
        # Handle both LinearFormKernel and generic LinearForm
        if isinstance(xp, LinearFormKernel):
            # Direct kernel representation - basis-free path
            x = self._inverse_mass_operator(xp.kernel)
            return x
        else:
            # Generic LinearForm - delegate to underlying Lebesgue space
            # The Lebesgue space's from_dual will extract the kernel
            # representation from the generic LinearForm
            kernel = self._underlying_space.from_dual(xp)

            # Apply inverse mass to convert from Lebesgue to Sobolev
            x = self._inverse_mass_operator(kernel)
            return x

    def _create_underlying_space(
        self,
        dim,
        function_domain,
        basis,
        integration_config,
        parallel_config,
    ) -> 'Lebesgue':
        from .lebesgue_space import Lebesgue
        return Lebesgue(
            dim,
            function_domain,
            basis=basis,
            integration_config=integration_config,
            parallel_config=parallel_config,
        )

    @property
    def mass_operator_factor(self):
        from .operators import BesselSobolev
        return BesselSobolev(
            self._underlying_space,
            self._underlying_space,
            k=self._k,
            s=self._s,
            L=self._L,
            dofs=self._dofs
        )

    @property
    def inverse_mass_operator_factor(self):
        from .operators import BesselSobolevInverse
        return BesselSobolevInverse(
            self._underlying_space,
            self._underlying_space,
            k=self._k,
            s=self._s,
            L=self._L,
            dofs=self._dofs
        )

    def restrict(self, restricted_space: "Sobolev", new_bcs=None):
        """
        Restrict Sobolev space to a subspace with new boundary conditions.

        This creates a new Sobolev space on a restricted domain by
        restricting the underlying Laplacian operator. The new space
        can have different boundary conditions at the new boundaries.

        Args:
            restricted_space: Target Sobolev space with restricted domain.
                Must have the same regularity parameters (s, k) as the
                original space. The function domain must be a subset of
                the original domain.
            new_bcs: New boundary conditions for the Laplacian operator.
                If None, uses the same boundary conditions as the original.

        Returns:
            New Sobolev space on the restricted domain.

        Example:
            >>> # Original Sobolev space on [0, 1]
            >>> M_full = Sobolev(...)
            >>> # Restrict to [0, 0.5] with new boundary conditions
            >>> M_lower = M_full.restrict(
            ...     restricted_space=Sobolev_lower,
            ...     new_bcs=BoundaryConditions(bc_type='mixed',
            ...                                left=0, right=None)
            ... )
        """
        # Validate regularity parameters match
        if not hasattr(restricted_space, '_s') or \
           restricted_space._s != self._s:
            raise ValueError(
                f"Regularity parameter s must match: "
                f"original={self._s}, restricted={restricted_space._s}"
            )
        if not hasattr(restricted_space, '_k') or \
           restricted_space._k != self._k:
            raise ValueError(
                f"Scaling parameter k must match: "
                f"original={self._k}, restricted={restricted_space._k}"
            )

        # Restrict the underlying Laplacian operator
        L_restricted = self._L.restrict(
            restricted_space._underlying_space,
            new_bcs=new_bcs
        )

        # Note: We don't create a new Sobolev space here,
        # we just update the Laplacian of the provided restricted_space
        # This is because the restricted_space is already constructed
        # with the correct domain and dimensions
        restricted_space._L = L_restricted

        # Update the mass operators with the new Laplacian
        from .operators import BesselSobolev, BesselSobolevInverse

        M_op = BesselSobolev(
            restricted_space._underlying_space,
            restricted_space._underlying_space,
            k=restricted_space._k,
            s=2 * restricted_space._s,
            L=L_restricted,
            dofs=L_restricted._dofs
        )
        M_op_inv = BesselSobolevInverse(
            restricted_space._underlying_space,
            restricted_space._underlying_space,
            k=restricted_space._k,
            s=2 * restricted_space._s,
            L=L_restricted,
            dofs=L_restricted._dofs
        )

        # Update the mass operators
        restricted_space._mass_operator = M_op
        restricted_space._inverse_mass_operator = M_op_inv

        return restricted_space

    @classmethod
    def with_discontinuities(
        cls,
        dim: int,
        function_domain: "IntervalDomain",
        discontinuity_points: list,
        s: float,
        k: float,
        bcs,
        alpha: float,
        *,
        basis: Optional[Union[str, list]] = None,
        dim_per_subspace: Optional[list] = None,
        basis_per_subspace: Optional[list] = None,
        bcs_per_subspace: Optional[list] = None,
        laplacian_method: str = 'spectral',
        dofs: int = 100,
        n_samples: int = 2048,
        integration_config = None,
    ) -> "SobolevSpaceDirectSum":
        """
        Create a SobolevSpaceDirectSum with discontinuities.

        This factory method creates a direct sum of Sobolev spaces, where
        each component space is defined on a subinterval separated by
        discontinuity points. This allows modeling of functions with jump
        discontinuities in Sobolev spaces.

        The total dimension `dim` is distributed across the subspaces.
        By default, dimension is allocated proportionally to subinterval
        lengths, but you can specify custom dimensions.

        Args:
            dim: Total dimension across all subspaces
            function_domain: The full interval domain
            discontinuity_points: List of points where discontinuities occur
            s: Sobolev regularity parameter
            k: Sobolev scaling parameter
            bcs: Default boundary conditions for Laplacian operators.
                Used for all subspaces unless bcs_per_subspace is provided.
            alpha: Laplacian scaling parameter
            basis: Basis type for ALL subspaces (string like 'fourier',
                'sine', etc., or 'none'). Ignored if basis_per_subspace
                is provided.
            dim_per_subspace: Optional list specifying dimension of each
                subspace. If None, dimensions are allocated proportionally
                to subinterval lengths. Must sum to `dim`.
            basis_per_subspace: Optional list of basis specifications, one
                for each subspace. Must have length equal to number of
                subspaces.
            bcs_per_subspace: Optional list of boundary conditions, one
                for each subspace. This allows specifying different boundary
                conditions at the discontinuities (e.g., DN for first subspace,
                ND for second). If None, the same `bcs` is used for all.
                Must have length equal to number of subspaces.
            laplacian_method: Method for Laplacian operators ('spectral', etc.)
            dofs: Number of degrees of freedom for Laplacian operators
            n_samples: Number of samples for spectral operators

        Returns:
            SobolevSpaceDirectSum representing the space with discontinuities

        Example:
            >>> from pygeoinf.interval.boundary_conditions import (
            ...     BoundaryConditions
            ... )
            >>> domain = IntervalDomain(0, 1)
            >>> bcs = BoundaryConditions(bc_type='dirichlet',
            ...                          left=0, right=0)
            >>> # Same boundary conditions for all subspaces
            >>> space = Sobolev.with_discontinuities(
            ...     100, domain, [0.5],
            ...     s=1, k=1, bcs=bcs, alpha=0.1
            ... )
            >>>
            >>> # Different boundary conditions per subspace
            >>> # DN on (0, 0.5), ND on (0.5, 1]
            >>> bcs_lower = BoundaryConditions(bc_type='dirichlet',
            ...                                left=0, right=None)
            >>> bcs_upper = BoundaryConditions(bc_type='neumann',
            ...                                left=None, right=0)
            >>> space = Sobolev.with_discontinuities(
            ...     100, domain, [0.5],
            ...     s=1, k=1, bcs=bcs,  # Default, will be overridden
            ...     alpha=0.1,
            ...     bcs_per_subspace=[bcs_lower, bcs_upper]
            ... )
        """
        from .lebesgue_space import Lebesgue
        from .operators import Laplacian

        # Validate basis parameter
        if isinstance(basis, list):
            raise ValueError(
                "Providing a list of basis functions is not supported for "
                "discontinuous spaces. Use basis_per_subspace to specify "
                "basis type for each subspace, or use basis='none' and set "
                "basis providers manually after creation."
            )

        # Split domain at discontinuities
        subdomains = function_domain.split_at_discontinuities(
            discontinuity_points
        )
        n_subspaces = len(subdomains)

        # Determine dimensions for each subspace
        if dim_per_subspace is None:
            # Allocate proportionally to subdomain lengths
            lengths = [sd.length for sd in subdomains]
            total_length = sum(lengths)

            # Start with proportional allocation (floored)
            dims = [int(dim * length / total_length)
                    for length in lengths]

            # Distribute remaining dimensions
            remainder = dim - sum(dims)
            # Add to largest subdomains first
            length_indices = sorted(range(n_subspaces),
                                    key=lambda i: lengths[i],
                                    reverse=True)
            for i in range(remainder):
                dims[length_indices[i % n_subspaces]] += 1
        else:
            # Use provided dimensions
            if len(dim_per_subspace) != n_subspaces:
                raise ValueError(
                    f"dim_per_subspace must have length {n_subspaces}, "
                    f"got {len(dim_per_subspace)}"
                )
            if sum(dim_per_subspace) != dim:
                raise ValueError(
                    f"dim_per_subspace must sum to {dim}, "
                    f"got {sum(dim_per_subspace)}"
                )
            dims = list(dim_per_subspace)

        # Determine basis for each subspace
        if basis_per_subspace is not None:
            # Use provided basis for each subspace
            if len(basis_per_subspace) != n_subspaces:
                raise ValueError(
                    f"basis_per_subspace must have length {n_subspaces}, "
                    f"got {len(basis_per_subspace)}"
                )
            bases = list(basis_per_subspace)
        else:
            # Use same basis for all subspaces
            bases = [basis] * n_subspaces

        # Determine boundary conditions for each subspace
        if bcs_per_subspace is not None:
            # Use provided boundary conditions for each subspace
            if len(bcs_per_subspace) != n_subspaces:
                raise ValueError(
                    f"bcs_per_subspace must have length {n_subspaces}, "
                    f"got {len(bcs_per_subspace)}"
                )
            bcs_list = list(bcs_per_subspace)
        else:
            # Use same boundary conditions for all subspaces
            bcs_list = [bcs] * n_subspaces

        # Create default integration config if not provided
        from .configs import IntegrationConfig
        if integration_config is None:
            integration_config = IntegrationConfig(
                method='simpson',
                n_points=1000
            )

        # Create subspaces with their own Laplacian operators
        subspaces = []
        for d, subdomain, b, bc in zip(dims, subdomains, bases, bcs_list):
            # Create underlying Lebesgue space for this subdomain
            M_lebesgue = Lebesgue(
                0, subdomain, basis=None,
                integration_config=integration_config
            )

            # Create Laplacian operator for this subdomain
            # This uses the new boundary conditions specific to this subspace
            laplacian = Laplacian(
                M_lebesgue,
                bc,  # Per-subspace boundary conditions
                alpha,
                method=laplacian_method,
                dofs=dofs,
                n_samples=n_samples,
                integration_config=integration_config
            )

            # Create Sobolev space on this subdomain
            # Pass configs to ensure consistency
            sobolev_subspace = cls(
                d,
                subdomain,
                s,
                k,
                laplacian,
                basis=b,
                integration_config=integration_config
            )
            subspaces.append(sobolev_subspace)

        # Return direct sum
        return SobolevSpaceDirectSum(subspaces)


class SobolevSpaceDirectSum(HilbertSpaceDirectSum):
    """
    Direct sum of Sobolev spaces using LinearFormKernel for basis-free operations.

    This class extends HilbertSpaceDirectSum to work with Sobolev spaces without
    requiring explicit basis functions. It uses LinearFormKernel which computes
    inner products via integration rather than component-based dot products.
    """

    def to_dual(self, xs: List[Function]) -> LinearFormKernel:
        """
        Maps a list of functions to a dual element using LinearFormKernel.

        For Sobolev spaces, this applies the mass operator to each component
        and wraps the result in a LinearFormKernel that uses integration
        instead of component-based operations.

        Args:
            xs: List of Function objects, one for each subspace

        Returns:
            LinearFormKernel that can evaluate inner products via integration
        """
        if len(xs) != self.number_of_subspaces:
            raise ValueError("Input list has incorrect number of vectors.")

        # Apply to_dual on each subspace (applies mass operator for Sobolev)
        # This returns a list of LinearFormKernel objects with mass-weighted kernels
        kernels = [space.to_dual(x).kernel for space, x in zip(self._spaces, xs)]

        # Get config from first subspace's underlying Lebesgue space
        # All subspaces should have the same config
        first_subspace = self._spaces[0]
        # Access via attribute to avoid type checker issues
        underlying = getattr(first_subspace, '_underlying_space', None)
        if underlying:
            int_cfg = underlying.integration.dual
            par_cfg = underlying.parallel.dual
        else:
            int_cfg = None
            par_cfg = None
        return LinearFormKernel(
            self,
            kernel=kernels,
            integration_config=int_cfg,
            parallel_config=par_cfg
        )

    def from_dual(self, xp: LinearFormKernel) -> List[Function]:
        """
        Maps a dual element back to a list of functions.

        Args:
            xp: LinearFormKernel containing kernel functions

        Returns:
            List of Function objects
        """
        # Handle both LinearFormKernel (specific) and generic LinearForm
        if isinstance(xp, LinearFormKernel):
            # The kernel is a list of mass-weighted functions
            # Apply from_dual to each to get back the original functions
            if isinstance(xp.kernel, list):
                # Get config from parent LinearFormKernel
                int_cfg = getattr(xp, 'integration_config', None)
                par_cfg = getattr(xp, 'parallel_config', None)
                return [
                    space.from_dual(
                        LinearFormKernel(
                            space, kernel=k,
                            integration_config=int_cfg,
                            parallel_config=par_cfg
                        )
                    ) for space, k in zip(self._spaces, xp.kernel)
                ]
            else:
                raise ValueError("Expected kernel to be a list for direct sum")
        else:
            # Delegate to base class for generic LinearForm objects
            return super().from_dual(xp)
