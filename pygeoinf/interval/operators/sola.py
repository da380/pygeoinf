"""SOLA operator for interval domains."""

from typing import Union, Optional, List, Callable, TYPE_CHECKING

import numpy as np

from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.linear_operators import LinearOperator

from ..lebesgue_space import Lebesgue
from ..sobolev_space import Sobolev
from ..functions import Function
from ..configs import IntegrationConfig
from ..function_providers import IndexedFunctionProvider
from ..linear_form_kernel import LinearFormKernel

if TYPE_CHECKING:
    from pygeoinf import LinearForm


class SOLAOperator(LinearOperator):
    """
    SOLA operator that applies kernel functions to input functions via
    integration.

    This operator takes a function from a Lebesgue space and computes integrals
    against a set of kernel functions, resulting in a vector in the specified
    Euclidean space.

    The operator maps: Lebesgue -> EuclideanSpace

    For each kernel function k_i, it computes: ∫ f(x) * k_i(x) dx

    The kernel functions can be provided in three ways:
    1. Via a FunctionProvider (original functionality)
    2. Via a list of Function objects
    3. Via a list of callables (automatically converted to Function objects)

    For direct sum domains (LebesgueSpaceDirectSum), use the static method
    `for_direct_sum` to create a RowLinearOperator that operates on each
    subspace independently.

    Examples:
        # Using a function provider
        >>> provider = NormalModesProvider(lebesgue_space)
        >>> sola_op = SOLAOperator(lebesgue_space, euclidean_space,
        ...                        function_provider=provider)

        # Using direct callables
        >>> kernels = [lambda x: np.sin(x), lambda x: np.cos(x)]
        >>> sola_op = SOLAOperator(lebesgue_space, euclidean_space,
        ...                        functions=kernels)

        # Using Function objects
        >>> func1 = Function(lebesgue_space, evaluate_callable=lambda x: x**2)
        >>> func2 = Function(lebesgue_space, evaluate_callable=lambda x: x**3)
        >>> sola_op = SOLAOperator(lebesgue_space, euclidean_space,
        ...                        functions=[func1, func2])

        # For direct sum spaces (discontinuous functions)
        >>> from pygeoinf.direct_sum import RowLinearOperator
        >>> direct_sum_space = Lebesgue.with_discontinuities(...)
        >>> provider = NormalModesProvider(direct_sum_space, ...)
        >>> sola_op = SOLAOperator.for_direct_sum(
        ...     direct_sum_space, euclidean_space, provider
        ... )
    """

    def __init__(
        self,
        domain: Union[Lebesgue, Sobolev],
        codomain: EuclideanSpace,
        kernels: Optional[
            Union[
                IndexedFunctionProvider,
                List[Union[Function, Callable]]
            ]
        ] = None,
        cache_kernels: bool = False,
        integration_config: IntegrationConfig = IntegrationConfig(method='simpson', n_points=1000),
    ):
        """
        Initialize the SOLA operator.

        Args:
            domain: Lebesgue instance (the function space)
            codomain: EuclideanSpace instance that defines the output dimension
            function_provider: Provider for generating kernels.
                              If None and functions is None, creates a default
                              NormalModesProvider.
            functions: List of Function instances or callables to use as
                      kernels. If provided, takes precedence over
                      function_provider. Callables will be converted to
                      Function instances.
            random_state: Random seed for reproducible function generation
            cache_functions: If True, cache kernels after first
                           access for faster repeated operations
            integration_config: Integration configuration

        Note:
            For direct sum domains (LebesgueSpaceDirectSum), use the
            `for_direct_sum` class method instead.
        """
        # Check if domain is a direct sum - if so, give helpful error
        from pygeoinf.direct_sum import HilbertSpaceDirectSum
        if isinstance(domain, HilbertSpaceDirectSum):
            raise TypeError(
                "SOLAOperator constructor does not directly support "
                "HilbertSpaceDirectSum domains. Use the class method "
                "SOLAOperator.for_direct_sum() instead."
            )

        self._domain = domain
        self._codomain = codomain
        self.N_d = codomain.dim
        self._kernels_provider = None
        self.cache_kernels = cache_kernels
        self._kernels_cache = {} if cache_kernels else None

        # Store integration config
        self.integration = integration_config

        self._initialize_kernels(kernels)

        super().__init__(
            domain,
            codomain,
            self._mapping,
            dual_mapping=self._dual_mapping
        )

    # Define the mapping function
    def _mapping(self, f: 'Function') -> np.ndarray:
        """Apply kernel functions to input function via integration."""
        return self._apply_kernels(f)

    def _dual_mapping(self, yp: 'LinearForm') -> 'LinearFormKernel':
        """Reconstruct function from data using kernel functions."""
        kernel = self._reconstruct_function(yp.components)
        return LinearFormKernel(self.domain, kernel=kernel, integration_config=self.integration)

    def _initialize_kernels(
        self,
        kernels: Optional[
            Union[
                IndexedFunctionProvider,
                List[Union[Function, Callable]]
            ]
        ] = None
    ):
        # Default to None - will use provider if set
        self._kernels = None

        if isinstance(kernels, list):
            if len(kernels) != self.N_d:
                raise ValueError(
                    f"Number of kernels ({len(kernels)}) must match "
                    f"codomain dimension ({self.N_d})"
                )
            if isinstance(kernels[0], Function):
                # Directly use provided Function instances
                self._kernels = kernels
            elif callable(kernels[0]):
                # Convert callables to Function instances
                self._kernels = [
                    Function(
                        self._domain,
                        evaluate_callable=func
                    )
                    for func in kernels
                ]
        elif isinstance(kernels, IndexedFunctionProvider):
            self._kernels_provider = kernels
            # _kernels already set to None above

    def get_kernel(self, index: int):
        """
        Lazily get the i-th kernel with optional caching.

        Args:
            index: Index of the kernel to retrieve

        Returns:
            Function: The i-th kernel
        """
        # If kernels are directly provided, return from list
        # Use getattr for backwards compatibility with old instances
        kernels = getattr(self, '_kernels', None)
        if kernels is not None:
            return kernels[index]

        # Otherwise use the provider to get the kernel
        assert self._kernels_provider is not None  # For type checker
        return self._kernels_provider.get_function_by_index(index)

    def _apply_kernels(self, func):
        """
        Apply the kernel functions to a function by integrating their product.

        For each kernel k_i, computes ∫ func(x) * k_i(x) dx

        Args:
            func: Function from the domain space

        Returns:
            numpy.ndarray: Vector of data in R^{N_d}
        """
        data = np.zeros(self.N_d)

        for i in range(self.N_d):
            # Lazily get the i-th kernel
            kernel = self.get_kernel(i)
            # Compute integral of product: ∫ func(x) * kernel(x) dx
            # Avoid creating intermediate Function to prevent deep recursion
            def product_callable(x):
                # Disable domain checking for kernel evaluation since:
                # 1. Kernel may be defined on larger domain than func
                # 2. Integration points are guaranteed to be in func's domain
                # 3. Boundary points are handled correctly by uniform_mesh
                return func.evaluate(x) * kernel.evaluate(x, check_domain=False)

            product_func = Function(self.domain, evaluate_callable=product_callable)
            data[i] = product_func.integrate(
                method=self.integration.method,
                n_points=self.integration.n_points
            )

        return data

    def _reconstruct_function(self, data):
        """
        Reconstruct a function from data using lazy evaluation.

        Args:
            data: numpy.ndarray of data in R^{N_d}

        Returns:
            Function: Reconstructed function in the domain space
        """
        # Collect non-zero terms to avoid deep recursion
        terms = []
        for i, coeff in enumerate(data):
            if abs(coeff) > 1e-14:  # Avoid numerical noise
                kernel = self.get_kernel(i)
                terms.append((coeff, kernel))

        # Create a single callable that evaluates all terms
        if not terms:
            return self.domain.zero

        def evaluate_sum(x):
            result = np.zeros_like(x) if isinstance(x, np.ndarray) else 0.0
            for coeff, kernel in terms:
                result = result + coeff * kernel.evaluate(x)
            return result

        return Function(self.domain, evaluate_callable=evaluate_sum)

    def get_kernels(self):
        """
        Get the list of kernels used by this operator.
        Note: This materializes all functions and may be expensive.

        Returns:
            list: List of kernels used for projection
        """
        return [self.get_kernel(i) for i in range(self.N_d)]

    def compute_gram_matrix(self):
        """
        Compute the Gram matrix of the kernels using function integration.

        For kernels k_i, k_j, computes ∫ k_i(x) * k_j(x) dx

        Returns:
            numpy.ndarray: N_d x N_d matrix of integrals between kernels
        """
        gram = np.zeros((self.N_d, self.N_d))

        for i in range(self.N_d):
            kernel_i = self.get_kernel(i)
            for j in range(self.N_d):
                kernel_j = self.get_kernel(j)
                # Compute integral: ∫ k_i(x) * k_j(x) dx
                # Avoid creating intermediate Function to prevent deep recursion

                def product_callable(x):
                    return kernel_i.evaluate(x) * kernel_j.evaluate(x)

                product_func = Function(
                    self.domain, evaluate_callable=product_callable
                )
                gram[i, j] = product_func.integrate(
                    method=self.integration.method,
                    n_points=self.integration.n_points
                )

        return gram

    def clear_cache(self):
        """Clear the function cache if caching is enabled."""
        if self.cache_functions and self._function_cache is not None:
            self._function_cache.clear()

    def get_cache_info(self):
        """
        Get information about the function cache.

        Returns:
            dict: Cache statistics including size and hit rate
        """
        if not self.cache_functions:
            return {"caching_enabled": False}

        assert self._function_cache is not None  # For type checker
        return {
            "caching_enabled": True,
            "cached_functions": len(self._function_cache),
            "total_functions": self.N_d,
            "cache_coverage": len(self._function_cache) / self.N_d
        }

    def __str__(self):
        """String representation of the SOLA operator."""
        provider_type = (type(self._kernels_provider).__name__
                        if self._kernels_provider else "direct functions")
        return (f"SOLAOperator: {self.domain} -> {self.codomain}\n"
                f"  Uses {self.N_d} kernels from {provider_type}\n"
                f"  Domain dimension: {self.domain.dim}\n"
                f"  Codomain dimension: {self.codomain.dim}")

    @staticmethod
    def for_direct_sum(
        domain,  # HilbertSpaceDirectSum
        codomain: EuclideanSpace,
        kernels: Union[
            IndexedFunctionProvider,
            List[Union[Function, Callable]]
        ],
        cache_kernels: bool = False,
        integration_config: IntegrationConfig = IntegrationConfig(method='simpson', n_points=1000),
    ):  # Returns RowLinearOperator
        """
        Create SOLAOperator for direct sum domain (discontinuous functions).

        This method creates a RowLinearOperator where each block operates
        on one of the subspaces of the direct sum. The kernel functions
        are used on each subdomain independently.

        Args:
            domain: HilbertSpaceDirectSum (e.g., LebesgueSpaceDirectSum)
            codomain: EuclideanSpace defining the output dimension
            kernels: Provider or list of kernels defined on the full domain
            cache_kernels: If True, cache kernels after first access
            integration_config: Integration configuration

        Returns:
            RowLinearOperator mapping from the direct sum space to the
            codomain by integrating against kernels on each subdomain.

        Example:
            >>> # Create a space with discontinuity
            >>> M = Lebesgue.with_discontinuities(
            ...     200, domain, [0.5], basis=None
            ... )
            >>> # Create kernels that span the full domain
            >>> provider = NormalModesProvider(M, ...)
            >>> # Create the operator
            >>> G = SOLAOperator.for_direct_sum(M, D, provider)
            >>> # G can act on discontinuous functions:
            >>> # G([f_lower, f_upper])
        """
        from pygeoinf.direct_sum import (
            HilbertSpaceDirectSum,
            RowLinearOperator
        )
        from ..lebesgue_space import Lebesgue as LebesgueSpace
        from ..function_providers import IndexedFunctionProvider

        if not isinstance(domain, HilbertSpaceDirectSum):
            raise TypeError(
                f"domain must be HilbertSpaceDirectSum, "
                f"got {type(domain)}"
            )

        # Create a SOLA operator for each subspace
        operators = []
        for i in range(domain.number_of_subspaces):
            subspace = domain.subspace(i)

            # Type assertion for type checker (LebesgueSpaceDirectSum
            # subspaces are Lebesgue instances)
            if not isinstance(subspace, (LebesgueSpace, Sobolev)):
                raise TypeError(
                    f"SOLAOperator requires Lebesgue or Sobolev subspaces,"
                    f" got {type(subspace)}"
                )

            # Restrict kernels to this subspace
            # If kernels is a provider, use restrict() method
            # If kernels is a list, restrict each function individually
            if isinstance(kernels, IndexedFunctionProvider):
                # Use provider restriction - this is the clean way!
                restricted_kernels = kernels.restrict(subspace)
            elif isinstance(kernels, list):
                # Restrict each function in the list
                restricted_kernels = []
                for kernel in kernels:
                    if isinstance(kernel, Function):
                        restricted_kernels.append(kernel.restrict(subspace))
                    elif callable(kernel):
                        # For callable, wrap in Function first
                        # This assumes the callable is defined on a larger
                        # domain
                        raise NotImplementedError(
                            "Cannot automatically restrict callable kernels."
                            " Please provide a FunctionProvider or "
                            "pre-restricted Functions."
                        )
                    else:
                        raise TypeError(f"Unknown kernel type: {type(kernel)}")
            else:
                raise TypeError(
                    f"kernels must be IndexedFunctionProvider or list, "
                    f"got {type(kernels)}"
                )

            # Create SOLAOperator with restricted kernels
            sola_sub = SOLAOperator(
                subspace,
                codomain,
                kernels=restricted_kernels,
                cache_kernels=cache_kernels,
                integration_config=integration_config
            )
            operators.append(sola_sub)

        # Create and return row operator
        return RowLinearOperator(operators)
