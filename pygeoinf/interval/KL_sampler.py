"""Spectral (KL) sampling utilities for interval operators.

Provides `SpectralSampler` for Gaussian measures with covariance admitting
eigenpairs (λ_i, φ_i). Generates samples using truncated expansion:

    u = m + Σ_{i<k} sqrt(λ_i) ξ_i φ_i,  ξ_i ~ N(0,1).

Also returns a covariance factor L so that C ≈ L L*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .operators import SpectralOperator
    from .functions import Function
from pygeoinf import EuclideanSpace, LinearOperator


@dataclass
class TruncationInfo:
    n_modes: int
    reason: str
    energy_fraction: Optional[float] = None


class KLSampler:
    """Sampler for Gaussian measures using a spectral (KL) expansion.

    Parameters
    ----------
    operator : object implementing get_eigenvalue(i) & get_eigenfunction(i)
        Typically `InverseLaplacian` or another covariance operator.
    mean : element of operator.domain, optional
        Mean function (defaults to zero element of the domain).
    n_modes : int, optional
        Explicit number of modes to retain. Mutually exclusive with
        `energy_tol`.
    energy_tol : float in (0,1], optional
        Retain the minimal number of leading modes whose cumulative
        eigenvalue sum fraction >= energy_tol (approximate if tail unknown).
    max_modes : int, optional
        Hard cap on modes (used with energy_tol to avoid runaway growth).
    rng : np.random.Generator, optional
        Random generator for reproducibility.
    cache : bool, default=True
        Cache fetched eigenpairs.
    """

    def __init__(
        self,
        operator: 'SpectralOperator',
        *,
        mean: Optional[Function] = None,
        n_modes: Optional[int] = None,
        energy_tol: Optional[float] = None,
        max_modes: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        cache: bool = True,
    ) -> None:

        if n_modes is not None and n_modes < 1:
            raise ValueError("n_modes must be positive")
        if energy_tol is not None:
            if not (0 < energy_tol <= 1):
                raise ValueError("energy_tol must be in (0,1]")
            if n_modes is not None:
                raise ValueError(
                    "Specify either n_modes or energy_tol, not both"
                )

        self._op = operator
        self._domain = operator.domain
        self._mean = mean if mean is not None else self._domain.zero
        self._explicit_n_modes = n_modes
        self._energy_tol = energy_tol
        self._max_modes = max_modes
        self._rng = rng if rng is not None else np.random.default_rng()
        self._cache_enabled = cache

        self._eigenvalues: List[float] = []
        self._eigenfunctions: List = []
        self._truncation: Optional[TruncationInfo] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _fetch_eigenpair(self, i: int):
        if i < len(self._eigenvalues):
            return self._eigenvalues[i], self._eigenfunctions[i]
        # Pull from operator
        lam = float(self._op.get_eigenvalue(i))
        if lam < 0:
            # Defensive clamp (should not happen for covariance)
            lam = 0.0
        func = self._op.get_eigenfunction(i)
        if self._cache_enabled:
            self._eigenvalues.append(lam)
            self._eigenfunctions.append(func)
        return lam, func

    def _determine_truncation(self) -> TruncationInfo:
        if self._truncation is not None:
            return self._truncation
        # Case 1: explicit number of modes
        if self._explicit_n_modes is not None:
            info = TruncationInfo(
                n_modes=self._explicit_n_modes, reason="explicit"
            )
            self._truncation = info
            # Ensure eigenpairs loaded
            for i in range(info.n_modes):
                self._fetch_eigenpair(i)
            return info
        # Case 2: energy-based
        if self._energy_tol is not None:
            # Heuristic: keep taking modes until relative drop large
            i = 0
            prev = None
            while True:
                lam, _ = self._fetch_eigenpair(i)
                i += 1
                if prev is not None and lam <= prev * (1 - self._energy_tol):
                    break
                prev = lam
                if self._max_modes is not None and i >= self._max_modes:
                    break
            info = TruncationInfo(
                n_modes=i, reason="energy_tol-heuristic", energy_fraction=None
            )
            self._truncation = info
            return info
        # Fallback: single mode
        info = TruncationInfo(n_modes=1, reason="default")
        self._truncation = info
        self._fetch_eigenpair(0)
        return info

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def truncation(self) -> TruncationInfo:
        return self._determine_truncation()

    @property
    def n_modes(self) -> int:
        return self.truncation.n_modes

    def eigenvalue(self, i: int) -> float:
        return self._fetch_eigenpair(i)[0]

    def eigenfunction(self, i: int):
        return self._fetch_eigenpair(i)[1]

    def covariance_factor(self) -> 'LinearOperator':
        """Return a covariance factor L : R^k → H with C ≈ L L*.

        Adjoint mapping implements L*: H → R^k via inner products with
        eigenfunctions divided by sqrt(λ_i) (assuming orthonormality).
        """
        k = self.n_modes
        eigvals = np.array([self.eigenvalue(i) for i in range(k)])
        eigfuncs = [self.eigenfunction(i) for i in range(k)]
        sqrt_lam = np.sqrt(eigvals)
        domain = EuclideanSpace(k)
        codomain = self._domain

        def mapping(w):  # w in R^k
            result = codomain.zero
            for coeff, lam_sqrt, phi in zip(w, sqrt_lam, eigfuncs):
                if coeff != 0.0:
                    # contribution sqrt(λ_i) * coeff * φ_i
                    result = codomain.add(result, (lam_sqrt * coeff) * phi)
            return result

        def adjoint_mapping(v):  # v in H
            comps = np.zeros(k)
            for i, (lam_sqrt, phi) in enumerate(zip(sqrt_lam, eigfuncs)):
                inner = self._domain.inner_product(phi, v)
                comps[i] = lam_sqrt * inner if lam_sqrt > 0 else 0.0
            return comps

        return LinearOperator(
            domain, codomain, mapping, adjoint_mapping=adjoint_mapping
        )

    def variance_function(self):
        """Return a function/vector representing the pointwise variance of the
        truncated KL expansion:

            Var[u](x) = Σ_{i<k} λ_i φ_i(x)^2

        The result is returned in the sampler's codomain (an element of the
        Hilbert space). For function spaces this is a `Function`; for
        Euclidean spaces this is a NumPy array.
        """
        k = self.n_modes
        codomain = self._domain

        # Special-case Euclidean spaces (component-wise variance)
        from pygeoinf.hilbert_space import EuclideanSpace

        if isinstance(codomain, EuclideanSpace):
            # Accumulate variance in component form
            var = np.zeros(codomain.dim)
            for i in range(k):
                lam = self.eigenvalue(i)
                phi = self.eigenfunction(i)
                # phi is expected to be a numpy array in EuclideanSpace
                phi_arr = np.asarray(phi)
                var += lam * (phi_arr ** 2)
            # Return as element of the space
            return codomain.from_components(var)

        # General function-space case: use function multiplication
        result = codomain.zero
        for i in range(k):
            lam = self.eigenvalue(i)
            phi = self.eigenfunction(i)
            # phi * phi should yield a Function in the same codomain
            phi_sq = phi * phi
            result = codomain.add(result, codomain.multiply(lam, phi_sq))

        return result

    def sample(self):
        """Draw a single sample (Function / element of domain)."""
        k = self.n_modes
        z = self._rng.standard_normal(k)
        eigvals = [self.eigenvalue(i) for i in range(k)]
        sample = self._domain.copy(self._mean)
        for i, lam in enumerate(eigvals):
            if lam <= 0:
                continue
            phi = self.eigenfunction(i)
            coeff = np.sqrt(lam) * z[i]
            sample = self._domain.add(sample, coeff * phi)
        return sample

    def samples(self, n: int):
        return [self.sample() for _ in range(n)]


__all__ = ["SpectralSampler", "TruncationInfo"]
