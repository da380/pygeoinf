"""
L² spaces on interval domains.

This module provides L² Hilbert spaces on intervals as the foundation
for more specialized function spaces like Sobolev spaces.
"""

import numpy as np
from typing import Optional


from pygeoinf.hilbert_space import HilbertSpace, LinearOperator
from pygeoinf.hilbert_space import LinearForm
from pygeoinf.interval.l2_functions import Function
from pygeoinf.interval.interval_domain import IntervalDomain
from pygeoinf.interval.providers import BasisProvider


class L2Space(HilbertSpace):
    """
    L² Hilbert space on an interval [a,b] with inner product
    ⟨u,v⟩ = ∫_a^b u(x)v(x) dx.

    This class provides the foundation for Sobolev spaces and manages:
    - L² inner product and norm via integration
    - Basis function creation and management (Fourier, etc.)
    - Function evaluation and coefficient transformations
    - Domain operations on intervals

    This serves as the base class for SobolevSpace.
    """

    def __init__(
        self,
        dim: int,
        function_domain: IntervalDomain,
        /,
        *,
        basis_type: Optional[str] = None,
        basis_callables: Optional[list] = None,
        basis_provider: Optional[BasisProvider] = None,
    ):
        """
        Args:
            dim (int): Dimension of the space.
            function_domain (IntervalDomain): Domain object with optional
                boundary conditions.
            basis_type (str, optional): Type of basis functions to auto-gen
                ('fourier', 'hat', 'hat_homogeneous'). Creates a lazy provider.
            basis_callables (list, optional): List of callable functions that
                will be converted to L2Function basis functions on this space.
                Solves the circular dependency problem.
            basis_provider (BasisProvider, optional): Custom basis provider
                for basis functions.

        Note:
            Exactly one of basis_type, basis_callables, or basis_provider
            must be provided. If none are provided, defaults to 'fourier'.

            The three options correspond to:
            1. basis_type: Auto-generate standard basis (Fourier, hat, etc.)
            2. basis_callables: User-provided callable functions
            3. basis_provider: Custom lazy provider implementation
        """
        self._dim = dim
        self._function_domain = function_domain
        self._gaussian_measure = None

        # Validate that exactly one basis option is provided
        basis_options = [basis_type, basis_callables, basis_provider]
        non_none_count = sum(1 for opt in basis_options if opt is not None)

        if non_none_count == 0:
            # Default to fourier if nothing specified
            basis_type = 'fourier'
        elif non_none_count > 1:
            raise ValueError(
                "Exactly one of basis_type, basis_callables, or "
                "basis_provider must be provided, but multiple were given"
            )

        # Handle the three basis options
        if basis_callables is not None:
            if len(basis_callables) != dim:
                raise ValueError(
                    f"basis_callables length ({len(basis_callables)}) "
                    f"must match dimension ({dim})"
                )
            self._basis_type = 'custom'
            # Convert callables to L2Function objects after space is created
            self._pending_callables = basis_callables
            self._basis_functions = None  # Will be created after init
            self._basis_provider = None
        elif basis_provider is not None:
            self._basis_type = 'custom_provider'
            self._basis_functions = None
            self._basis_provider = basis_provider
            self._pending_callables = None
        else:
            # basis_type is specified (or defaulted to 'fourier')
            self._basis_type = basis_type
            self._basis_functions = None
            self._basis_provider = None  # Will be created below
            self._pending_callables = None

        # Create basis provider for standard basis types
        if basis_type in ['fourier', 'hat', 'hat_homogeneous']:
            from .providers import create_basis_provider
            self._basis_provider = create_basis_provider(self, basis_type)

        # Initialize Gram matrix as None - computed lazily when needed
        self._gram_matrix = None

        # Initialize the parent HilbertSpace with L² inner product
        super().__init__(
            dim,
            self._to_components,
            self._from_components,
            self.inner_product,
            self._default_to_dual,
            self._default_from_dual,
            copy=self._copy,
        )

        # to Function objects (this solves the circular dependency)
        if basis_callables:
            l2_funcs = []
            for callable_func in basis_callables:
                l2_func = Function(self, evaluate_callable=callable_func)
                l2_funcs.append(l2_func)
            self._manual_basis_functions = l2_funcs

    @property
    def dim(self):
        """Return the dimension of the space."""
        return self._dim

    @property
    def function_domain(self):
        """Return the IntervalDomain object for this space."""
        return self._function_domain

    def get_basis_function(self, index: int):
        """Get basis function by index, works with both lazy and explicit."""
        if self._basis_functions is not None:
            return self._basis_functions[index]
        elif self._basis_provider is not None:
            return self._basis_provider.get_basis_function(index)
        else:
            raise RuntimeError(
                "Neither explicit nor lazy basis functions available"
            )

    @property
    def basis_functions(self):
        """Property to access basis functions with consistent interface."""
        if self._basis_functions is not None:
            return self._basis_functions
        else:
            # Use the lazy provider to get all basis functions as a list
            # This ensures consistent interface - always returns a list
            return self._basis_provider.get_all_basis_functions()

    def basis_function(self, i):
        """Return the ith basis function directly."""
        if i < 0 or i >= self.dim:
            raise IndexError(f"Basis index {i} out of range [0, {self.dim})")

        return self.get_basis_function(i)

    @property
    def basis_provider(self):
        """Return the lazy basis provider for this space."""
        if self._basis_provider is not None:
            return self._basis_provider
        else:
            raise RuntimeError("No basis provider available for this space")

    @property
    def gram_matrix(self):
        """The Gram matrix of basis functions."""
        if self._gram_matrix is None:
            self._compute_gram_matrix()
        return self._gram_matrix

    @property
    def basis_type(self):
        """The type of basis functions used."""
        return self._basis_type

    def inner_product(self, u, v):
        """
        L² inner product: ⟨u,v⟩_L² = ∫_a^b u(x)v(x) dx

        Args:
            u, v: Functions in this L² space

        Returns:
            float: L² inner product

        For L² functions, we compute ⟨u,v⟩_L² = ∫_a^b u(x)v(x) dx through
        numerical integration, not pointwise evaluation (which is not
        mathematically well-defined for general L² functions).
        """
        # For L² functions, we need to be careful about pointwise operations
        # In practice, we work with smooth approximations
        product = u * v
        return product.integrate()

    def _compute_gram_matrix(self):
        """
        Compute the Gram matrix of the basis functions using L2 inner products.
        """
        n = self.dim
        self._gram_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):  # Only compute upper triangle
                # Get basis functions - works with both lazy and explicit
                basis_i = self.get_basis_function(i)
                basis_j = self.get_basis_function(j)

                inner_prod = self.inner_product(basis_i, basis_j)
                self._gram_matrix[i, j] = inner_prod
                self._gram_matrix[j, i] = inner_prod  # Symmetric matrix

    def project(self, f):
        """
        Project a function onto this L2 space.

        Args:
            f: Function to project (callable or L2Function)

        Returns:
            L2Function: The projection of f onto this space
        """
        if callable(f):
            # Create Function from callable
            func = Function(self, evaluate_callable=f)
        else:
            func = f

        # Compute coefficients via L2 inner products
        coeffs = self._to_components(func)
        return self._from_components(coeffs)

    def _to_components(self, u):
        """
        Convert a function to coefficients using inner products with basis
        functions.
        """
        # Compute right-hand side: b_i = <u, φ_i>_L²
        rhs = np.zeros(self.dim)
        for k in range(self.dim):
            basis_func = self.get_basis_function(k)
            rhs[k] = self.inner_product(u, basis_func)

        # Solve the linear system: G * c = rhs
        gram = self.gram_matrix
        if gram is None:
            raise ValueError("Gram matrix not computed")
        coeffs = np.linalg.solve(gram, rhs)
        return coeffs

    def _from_components(self, coeff):
        """
        Convert coefficients to a function using linear combination of
        basis functions.
        """
        coeff = np.asarray(coeff)
        if len(coeff) != self.dim:
            raise ValueError(f"Coefficients must have length {self.dim}")

        # Create Function directly with coefficients
        return Function(self, coefficients=coeff)

    # Default dual space mappings
    def _default_to_dual(self, u: Function):
        """Default mapping to dual space using Gram matrix."""
        return LinearForm(self, mapping=lambda v: self.inner_product(u, v))

    def _default_from_dual(self, up: LinearForm):
        """Default mapping from dual space using inverse Gram matrix."""
        dual_components = np.zeros(self.dim)
        for i in range(self.dim):
            basis_func = self.get_basis_function(i)
            dual_components[i] = up(basis_func)

        gram = self.gram_matrix
        if gram is None:
            raise ValueError("Gram matrix not computed")
        components = np.linalg.solve(gram, dual_components)
        return Function(
            self,
            coefficients=components,
        )

    def _copy(self, x):
        """Custom copy implementation for Functions."""
        return Function(
            self,
            coefficients=self.to_components(x).copy(),
            name=getattr(x, 'name', None)
        )

    # ========================================================================
    # Gaussian Measure Methods
    # ========================================================================

    @property
    def gaussian_measure(self):
        """
        Property to access the Gaussian measure for this L2 space.
        This is a convenience method that allows using the same interface
        for both KL and SPDE methods.
        """
        if self._gaussian_measure is None:
            raise ValueError("Gaussian measure not created yet. "
                             "Use create_gaussian_measure() first.")
        return self._gaussian_measure

    def gaussian_measure_kl(self, covariance: LinearOperator,
                            expectation: Optional[Function] = None,
                            kl_expansion: int = 100):
        """
        Create a Gaussian measure using Karhunen-Loève expansion.

        For a Gaussian measure μ = N(m, C), samples are generated as:
        X = m + Σⱼ₌₁^N √λⱼ ξⱼ φⱼ

        where {(λⱼ, φⱼ)} are eigenpairs of the covariance operator C,
        and ξⱼ ~ N(0,1) are i.i.d. standard normal variables.

        Args:
            covariance (LinearOperator): Covariance operator that should have
                a spectrum provider available. If the operator has a
                spectrum_provider attribute, it will be used for KL expansion.
                Otherwise, raises an error.
            expectation (Function, optional): Mean function. Defaults to zero.
            kl_expansion (int): Number of modes for KL expansion (default 100).

        Returns:
            GaussianMeasure: Gaussian measure with spectral covariance

        Example:
            >>> # Using LaplacianInverseOperator with spectrum
            >>> lap_inv = LaplacianInverseOperator(l2_space, bc)
            >>> measure = l2_space.gaussian_measure_kl(lap_inv)
            >>> sample = measure.sample()
        """
        from pygeoinf.gaussian_measure import GaussianMeasure

        # Check if the operator has a spectrum provider
        if not hasattr(covariance, 'spectrum_provider'):
            raise ValueError(
                f"Covariance operator {type(covariance).__name__} does not "
                f"have a spectrum_provider attribute. KL expansion requires "
                f"analytical spectrum information."
            )

        def sample_gaussian_kl():
            """
            Sample from Gaussian measure using Karhunen-Loève expansion.
            """
            # Ensure kl_expansion doesn't exceed space dimension
            effective_expansion = min(kl_expansion, self.dim)

            # Get eigenvalues using the spectrum provider
            eigenvalues = covariance.get_all_eigenvalues(n=effective_expansion)
            eigenvalues = np.array(eigenvalues)
            sqrt_eigenvalues = np.sqrt(np.maximum(eigenvalues, 1e-12))

            # Generate i.i.d. standard normal variables ξⱼ
            xi = np.random.randn(len(eigenvalues))

            # Compute coefficients: cⱼ = √λⱼ ξⱼ
            coefficients = sqrt_eigenvalues * xi

            # Create sample as linear combination: Σⱼ cⱼ φⱼ
            for j in range(len(eigenvalues)):
                basis_func = covariance.get_eigenfunction(j)
                if j == 0:
                    sample = coefficients[j] * basis_func
                else:
                    sample = sample + coefficients[j] * basis_func

            # Add expectation (mean)
            if expectation is not None:
                sample = sample + expectation

            return sample

        self._gaussian_measure = GaussianMeasure(
            covariance=covariance,
            expectation=expectation,
            sample=sample_gaussian_kl
        )

    def gaussian_measure_spde(self, precision_operator, expectation=None):
        """
        Create a Gaussian measure using SPDE/precision operator approach.

        For precision operator A = C^(-1), samples are generated by solving:
        A^(1/2) u = ξ

        where ξ is Gaussian white noise. This avoids explicit
        eigendecomposition and scales well with modern linear solvers.

        Args:
            precision_operator (LinearOperator): Precision operator A = C^(-1).
                Should map from this L2Space to some target space
                (often Sobolev). Must be self-adjoint and positive definite.
            expectation (Function, optional): Mean function. Defaults to zero.

        Returns:
            GaussianMeasure: Gaussian measure with operator-based covariance

        Example:
            >>> # Using Laplacian inverse as covariance
            >>> laplacian_inv = LaplacianInverseOperator(
            ...     sobolev_space, l2_space
            ... )
            >>> measure = l2_space.gaussian_measure_spde(laplacian_inv)
            >>> sample = measure.sample_spde()
        """
        from pygeoinf.gaussian_measure import GaussianMeasure

        # For SPDE approach, we store the precision operator directly
        # The actual sampling will be done via custom method
        return GaussianMeasure(
            covariance=precision_operator,
            expectation=expectation
        )

    def sample_gaussian_spde(
        self, precision_operator, expectation=None, n_samples=1
    ):
        """
        Sample from Gaussian measure using SPDE/precision operator approach.

        Solves the SPDE: A^(1/2) u = ξ where A is the precision operator.
        In practice, this often means: u = A^(-1) ξ
        (apply covariance to noise).

        Args:
            precision_operator (LinearOperator): Precision operator A = C^(-1)
            expectation (Function, optional): Mean function. Defaults to zero.
            n_samples (int): Number of samples to generate. Default 1.

        Returns:
            Function or list[Function]: Generated sample(s)

        Note:
            This assumes precision_operator actually represents the covariance
            operator C (i.e., A^(-1)), not the precision operator A itself.
            This is consistent with the LaplacianInverseOperator usage.
        """
        samples = []

        for _ in range(n_samples):
            # Generate white noise ξ in the domain space
            noise_coeffs = np.random.randn(self.dim)
            white_noise = self.from_components(noise_coeffs)

            # Apply covariance operator: u = C ξ
            # Note: precision_operator is actually the covariance operator here
            sample = precision_operator(white_noise)

            # Add expectation (mean)
            if expectation is not None:
                # Need to ensure both sample and expectation are in
                # the same space
                if hasattr(sample, 'space') and hasattr(expectation, 'space'):
                    if sample.space == expectation.space:
                        sample = sample.space.add(sample, expectation)
                    else:
                        # Handle case where sample is in codomain,
                        # expectation in domain
                        # This would need more sophisticated handling
                        pass

            samples.append(sample)

        return samples[0] if n_samples == 1 else samples

    def create_gaussian_measure(self, method='kl', kl_expansion: int = 100,
                                **kwargs):
        """
        Unified interface for creating Gaussian measures with different
        methods.

        Args:
            method (str): Sampling method - 'kl' for Karhunen-Loève,
                         'spde' for SPDE/precision operator
            kl_expansion (int): Number of modes for KL expansion (default 100)
            **kwargs: Method-specific arguments

        For method='kl':
            covariance (LinearOperator): Covariance operator that should have
                a spectrum provider available (e.g., LaplacianInverseOperator)
            expectation (Function, optional): Mean function

        For method='spde':
            precision_operator (LinearOperator): Precision/covariance operator
            expectation (Function, optional): Mean function

        Returns:
            GaussianMeasure: Configured Gaussian measure

        Example:
            >>> # Karhunen-Loève method
            >>> lap_inv = LaplacianInverseOperator(l2_space, bc)
            >>> measure1 = l2_space.create_gaussian_measure(
            ...     method='kl',
            ...     covariance=lap_inv
            ... )
            >>>
            >>> # SPDE method
            >>> measure2 = l2_space.create_gaussian_measure(
            ...     method='spde',
            ...     precision_operator=laplacian_inv
            ... )
        """
        if method == 'kl':
            self.gaussian_measure_kl(kl_expansion=kl_expansion, **kwargs)
        elif method == 'spde':
            self.gaussian_measure_spde(**kwargs)
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'kl' or 'spde'.")
