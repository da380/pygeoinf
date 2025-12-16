"""
Configuration dataclass for PLI (Probabilistic Linear Inference) experiments.

This module defines the configuration parameters for the PLI inference problem
demonstrated in pli.ipynb, which includes:
- Model space: L² function space with specified basis
- Data space: Euclidean space with sensitivity kernel observations
- Property space: Euclidean space with target kernel properties
- Prior: Gaussian measure with Bessel-Sobolev or inverse Laplacian covariance
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Literal
import json
from pathlib import Path


@dataclass
class BoundaryConditionConfig:
    """Configuration for boundary conditions.

    Supports three types:
    - 'dirichlet': Fixed value at boundaries (u(a) = left, u(b) = right)
    - 'neumann': Fixed derivative at boundaries (u'(a) = left, u'(b) = right)
    - 'robin': Linear combination (α*u + β*u' = 0)
    """
    bc_type: Literal['dirichlet', 'neumann', 'robin'] = 'neumann'
    left: Any = 0  # Value or dict for robin: {'alpha': a, 'beta': b}
    right: Any = 0  # Value or dict for robin: {'alpha': a, 'beta': b}


@dataclass
class IntegrationSettings:
    """Configuration for numerical integration.

    Attributes:
        method: Integration method ('simpson', 'trapz', 'quad')
        n_points: Number of integration points
    """
    method: str = 'simpson'
    n_points: int = 1000


@dataclass
class ParallelSettings:
    """Configuration for parallel computation.

    Attributes:
        enabled: Whether to use parallel processing
        n_jobs: Number of parallel jobs (-1 = use all cores)
    """
    enabled: bool = True
    n_jobs: int = 16


@dataclass
class PLIConfig:
    """Configuration for PLI (Probabilistic Linear Inference) experiments.

    This dataclass encapsulates all parameters needed to run the PLI
    inference problem, organized into several categories:

    Space Parameters:
        N: Model space dimension (number of basis functions)
        N_d: Number of data points (sensitivity kernels)
        N_p: Number of property points (target kernels)
        basis: Basis type ('sine', 'cosine', 'fourier', 'legendre')
        domain_a: Left boundary of function domain
        domain_b: Right boundary of function domain

    Prior Parameters (Bessel-Sobolev: C_0 = (k²I + αL)^(-s)):
        prior_type: 'bessel_sobolev' or 'inverse_laplacian'
        s: Sobolev order (smoothness parameter, s > 0.5)
        length_scale: Correlation length scale
        overall_variance: Prior variance scaling
        K: Number of KL modes for prior sampling

    Laplacian Parameters:
        alpha: Laplacian scaling parameter
        method: Method for Laplacian ('spectral' or 'fd')
        dofs: Degrees of freedom for spectral method
        n_samples: Number of samples for fast transforms

    Boundary Conditions:
        bc_type: 'dirichlet', 'neumann', or 'robin'
        bc_left: Left boundary value
        bc_right: Right boundary value

    Data Generation:
        noise_level: Relative noise level (fraction of signal max)
        random_seed: Random seed for reproducibility

    Forward Operator (Sensitivity Kernels):
        n_modes_range: Range of normal modes for kernel generation
        coeff_range: Coefficient range for kernel generation
        gaussian_width_percent_range: Gaussian width range (%)
        freq_range: Frequency range for oscillations

    Target Operator (Property Kernels):
        target_width: Width of bump functions for property extraction

    Computation Settings:
        compute_model_posterior: Whether to compute full model posterior
        integration: Integration configuration
        parallel: Parallelization configuration

    Experiment Metadata:
        name: Descriptive name for this configuration
        description: Longer description of experiment purpose
    """

    # Space parameters
    N: int = 100
    N_d: int = 50
    N_p: int = 20
    basis: str = 'cosine'
    domain_a: float = 0.0
    domain_b: float = 1.0

    # Prior parameters (Bessel-Sobolev)
    prior_type: Literal['bessel_sobolev', 'inverse_laplacian'] = 'bessel_sobolev'
    s: float = 3.0  # Sobolev order
    length_scale: float = 0.5  # Correlation length scale
    overall_variance: float = 1.0  # Prior variance
    K: int = 100  # KL modes

    # Laplacian parameters
    alpha: Optional[float] = None  # Computed from length_scale if None
    method: str = 'spectral'
    dofs: int = 100
    n_samples: int = 2048
    use_fast_transforms: bool = True

    # Boundary conditions
    bc_type: str = 'neumann'
    bc_left: Any = 0
    bc_right: Any = 0

    # Data generation
    noise_level: float = 0.1
    random_seed: int = 42

    # Forward operator (sensitivity kernels)
    n_modes_range: tuple = (1, 50)
    coeff_range: tuple = (-5, 5)
    gaussian_width_percent_range: tuple = (1, 5)
    freq_range: tuple = (0.1, 20)

    # Target operator (property kernels)
    target_width: float = 0.2

    # Computation settings
    compute_model_posterior: bool = False
    integration_method: str = 'simpson'
    integration_n_points: int = 1000
    parallel_enabled: bool = True
    parallel_n_jobs: int = 16

    # Experiment metadata
    name: str = "pli_default"
    description: str = "Default configuration for PLI inference"

    def __post_init__(self):
        """Validate and derive additional parameters."""
        # Validate positive values
        if self.N <= 0:
            raise ValueError(f"N must be positive, got {self.N}")
        if self.N_d <= 0:
            raise ValueError(f"N_d must be positive, got {self.N_d}")
        if self.N_p <= 0:
            raise ValueError(f"N_p must be positive, got {self.N_p}")
        if self.K <= 0:
            raise ValueError(f"K must be positive, got {self.K}")

        # Validate smoothness parameter
        if self.s <= 0.5:
            raise ValueError(f"s must be > 0.5 for convergence, got {self.s}")

        # Validate length scale
        if self.length_scale <= 0:
            raise ValueError(f"length_scale must be positive, got {self.length_scale}")

        # Validate variance
        if self.overall_variance <= 0:
            raise ValueError(f"overall_variance must be positive, got {self.overall_variance}")

        # Validate noise level
        if self.noise_level < 0:
            raise ValueError(f"noise_level must be non-negative, got {self.noise_level}")

        # Validate domain
        if self.domain_a >= self.domain_b:
            raise ValueError(f"domain_a must be < domain_b, got [{self.domain_a}, {self.domain_b}]")

        # Validate basis type
        valid_bases = ['sine', 'cosine', 'fourier', 'legendre']
        if self.basis not in valid_bases:
            raise ValueError(f"basis must be one of {valid_bases}, got {self.basis}")

        # Validate prior type
        if self.prior_type not in ['bessel_sobolev', 'inverse_laplacian']:
            raise ValueError(f"prior_type must be 'bessel_sobolev' or 'inverse_laplacian'")

        # Validate boundary condition type
        if self.bc_type not in ['dirichlet', 'neumann', 'robin']:
            raise ValueError(f"bc_type must be 'dirichlet', 'neumann', or 'robin'")

    @property
    def k(self) -> float:
        """Compute Bessel parameter k from overall variance and smoothness."""
        return pow(self.overall_variance, -0.5 / self.s)

    @property
    def alpha_computed(self) -> float:
        """Compute Laplacian scaling from length scale and k."""
        if self.alpha is not None:
            return self.alpha
        return (self.length_scale ** 2) * (self.k ** 2)

    @property
    def domain_length(self) -> float:
        """Length of the function domain."""
        return self.domain_b - self.domain_a

    @property
    def domain_center(self) -> float:
        """Center of the function domain."""
        return (self.domain_a + self.domain_b) / 2

    @property
    def target_centers(self):
        """Compute centers for target bump functions."""
        import numpy as np
        return np.linspace(
            self.domain_a + self.target_width / 2,
            self.domain_b - self.target_width / 2,
            self.N_p
        )

    @property
    def bc_config(self) -> Dict[str, Any]:
        """Get boundary condition configuration as dict."""
        return {
            'bc_type': self.bc_type,
            'left': self.bc_left,
            'right': self.bc_right
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        d = asdict(self)
        # Add derived parameters
        d['k'] = self.k
        d['alpha_computed'] = self.alpha_computed
        d['domain_length'] = self.domain_length
        d['domain_center'] = self.domain_center
        return d

    def save(self, path: Path) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> 'PLIConfig':
        """Load configuration from JSON file."""
        path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)

        # Remove derived parameters if present
        for key in ['k', 'alpha_computed', 'domain_length', 'domain_center', 'target_centers']:
            data.pop(key, None)

        # Convert tuples from lists
        for key in ['n_modes_range', 'coeff_range', 'gaussian_width_percent_range', 'freq_range']:
            if key in data and isinstance(data[key], list):
                data[key] = tuple(data[key])

        return cls(**data)

    def copy(self, **changes) -> 'PLIConfig':
        """Create a copy of this configuration with specified changes."""
        d = self.to_dict()
        # Remove derived parameters
        for key in ['k', 'alpha_computed', 'domain_length', 'domain_center', 'target_centers']:
            d.pop(key, None)
        d.update(changes)

        # Convert tuples from lists if needed
        for key in ['n_modes_range', 'coeff_range', 'gaussian_width_percent_range', 'freq_range']:
            if key in d and isinstance(d[key], list):
                d[key] = tuple(d[key])

        return PLIConfig(**d)


# =============================================================================
# PRE-DEFINED CONFIGURATIONS
# =============================================================================

def get_fast_config() -> PLIConfig:
    """Fast configuration for quick testing (reduced resolution)."""
    return PLIConfig(
        N=50,
        N_d=25,
        N_p=10,
        K=30,
        dofs=50,
        n_samples=512,
        integration_n_points=500,
        compute_model_posterior=False,
        name="fast_test",
        description="Fast configuration for testing with reduced resolution"
    )


def get_standard_config() -> PLIConfig:
    """Standard configuration matching pli.ipynb defaults."""
    return PLIConfig(
        N=100,
        N_d=50,
        N_p=20,
        K=100,
        basis='cosine',
        s=3.0,
        length_scale=0.5,
        overall_variance=1.0,
        noise_level=0.1,
        compute_model_posterior=False,
        parallel_enabled=True,
        parallel_n_jobs=16,
        name="standard",
        description="Standard configuration matching pli.ipynb"
    )


def get_high_resolution_config() -> PLIConfig:
    """High resolution configuration for detailed analysis."""
    return PLIConfig(
        N=200,
        N_d=100,
        N_p=40,
        K=200,
        dofs=200,
        n_samples=4096,
        integration_n_points=2000,
        compute_model_posterior=False,
        parallel_enabled=True,
        parallel_n_jobs=-1,
        name="high_resolution",
        description="High resolution configuration for detailed analysis"
    )


def get_full_posterior_config() -> PLIConfig:
    """Configuration with full model posterior computation."""
    return PLIConfig(
        N=100,
        N_d=50,
        N_p=20,
        K=100,
        compute_model_posterior=True,
        parallel_enabled=True,
        parallel_n_jobs=-1,
        name="full_posterior",
        description="Configuration with full model posterior computation for sampling"
    )


def get_smooth_prior_config() -> PLIConfig:
    """Configuration with smoother prior (higher s, longer length scale)."""
    return PLIConfig(
        N=100,
        N_d=50,
        N_p=20,
        K=100,
        s=4.0,  # Higher smoothness
        length_scale=0.7,  # Longer correlation
        overall_variance=1.0,
        name="smooth_prior",
        description="Configuration with smoother prior for regularization"
    )


def get_rough_prior_config() -> PLIConfig:
    """Configuration with rougher prior (lower s, shorter length scale)."""
    return PLIConfig(
        N=100,
        N_d=50,
        N_p=20,
        K=100,
        s=1.5,  # Lower smoothness (minimum stable is ~1)
        length_scale=0.2,  # Shorter correlation
        overall_variance=1.0,
        name="rough_prior",
        description="Configuration with rougher prior allowing more variation"
    )


def get_low_noise_config() -> PLIConfig:
    """Configuration with low noise for high-SNR scenarios."""
    return PLIConfig(
        N=100,
        N_d=50,
        N_p=20,
        K=100,
        noise_level=0.01,
        name="low_noise",
        description="Low noise configuration for high-SNR testing"
    )


def get_high_noise_config() -> PLIConfig:
    """Configuration with high noise for robustness testing."""
    return PLIConfig(
        N=100,
        N_d=50,
        N_p=20,
        K=100,
        noise_level=0.3,
        name="high_noise",
        description="High noise configuration for robustness testing"
    )
