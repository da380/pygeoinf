"""Robin boundary condition function provider."""

import math
from typing import TYPE_CHECKING
from .base import IndexedFunctionProvider
from ..utils.robin_utils import RobinRootFinder

if TYPE_CHECKING:
    from ..functions import Function


class RobinFunctionProvider(IndexedFunctionProvider):
    """
    Robin eigenfunctions for -d2/dx2 on (a,b) with separated BCs:
      alpha_0 u(a) + beta_0 u'(a) = 0,
      alpha_L u(b) + beta_L u'(b) = 0.

    Each eigenfunction has the form A cos(μ y) + B sin(μ y), y=x-a,
    where μ>0 satisfies the characteristic equation
      D(μ) = (alpha_0 alpha_L + beta_0 beta_L μ^2) sin(μL)
             + μ(alpha_0 beta_L − beta_0 alpha_L) cos(μL) = 0.

    This provider:
    - finds μ_k by bisection (robust) with one root per interval (~kπ/L,(k+1)π/L),
    - constructs (A,B) from the left BC (alpha_0 A + beta_0 μ B = 0),
    - normalizes φ_k in L²(a,b) numerically (Simpson) and caches.
    """
    def __init__(self, space,
                 bcs,
                 integration_method: str = 'simpson',
                 n_points: int = 2000,
                 root_tol: float = 1e-12,
                 max_bisect_iter: int = 100):
        super().__init__(space)
        self.alpha0 = float(bcs.get_parameter('left_alpha'))
        self.beta0 = float(bcs.get_parameter('left_beta'))
        self.alphaL = float(bcs.get_parameter('right_alpha'))
        self.betaL = float(bcs.get_parameter('right_beta'))
        self.value0 = float(bcs.get_parameter('left_value'))
        self.valueL = float(bcs.get_parameter('right_value'))

        self.integration_method = integration_method
        self.n_points = n_points
        self.root_tol = root_tol
        self.max_bisect_iter = max_bisect_iter

        self._mu_cache: list[float] = []     # μ_k roots
        self._func_cache: dict[int, 'Function'] = {}  # index -> Function

    # -- public API --
    def get_function_by_index(self, index: int, **kwargs) -> 'Function':
        if index < 0:
            raise ValueError("index must be ≥ 0")
        if index in self._func_cache:
            return self._func_cache[index]

        import numpy as np
        from ..functions import Function
        a, b = self.domain.a, self.domain.b
        L = b - a

        # get μ_k
        mu = self._mu_at(index)

        # Special pure Neumann case (alpha_0=alpha_L=0) -> μ0=0 gives constant mode
        if mu == 0.0:
            def const(x):
                return (np.ones_like(x) if isinstance(x, np.ndarray) else 1.0) / math.sqrt(L)
            f0 = Function(self.space, evaluate_callable=const, name="robin_constant")
            self._func_cache[index] = f0
            return f0

        # Build (A,B) from left BC alpha_0 A + beta_0 μ B = 0.
        A, B = self._coefficients_from_left_bc(mu)
        # raw eigenfunction (unnormalized)
        def raw(x):
            import numpy as np
            y = np.asarray(x) - a
            return A*np.cos(mu*y) + B*np.sin(mu*y)

        # normalize in L²(a,b)
        raw_func = Function(self.space, evaluate_callable=raw)
        norm2 = (raw_func * raw_func).integrate(method=self.integration_method,
                                                n_points=self.n_points)
        c = 1.0 / math.sqrt(max(norm2, 1e-300))

        def phi(x):
            return c * raw(x)

        f = Function(self.space, evaluate_callable=phi,
                     name=f"robin_mu={mu:.8g}")
        self._func_cache[index] = f
        return f

    # -- μ_k management --
    def _mu_at(self, k: int) -> float:
        while len(self._mu_cache) <= k:
            self._append_next_mu()
        return self._mu_cache[k]

    def _append_next_mu(self):
        # Compute next eigenvalue index
        index = len(self._mu_cache)

        # Use shared RobinRootFinder utility
        mu = RobinRootFinder.compute_robin_eigenvalue(
            index,
            self.alpha0,
            self.beta0,
            self.alphaL,
            self.betaL,
            self.domain.length,
            tol=self.root_tol,
            maxit=self.max_bisect_iter
        )
        self._mu_cache.append(mu)

    # -- build (A,B) from left BC --
    def _coefficients_from_left_bc(self, mu: float) -> tuple[float, float]:
        # Use shared RobinRootFinder utility
        return RobinRootFinder.compute_coefficients_from_left_bc(
            mu,
            self.alpha0,
            self.beta0,
            self.alphaL,
            self.betaL,
            self.domain.length
        )
