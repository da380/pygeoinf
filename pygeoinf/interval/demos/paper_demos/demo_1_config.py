"""
Configuration dataclass for Demo 1 experiments.

This module defines the configuration parameters for the multi-component
Bayesian inference problem demonstrated in demo_1.ipynb, which includes
three function spaces (vp, vs, rho) and two Euclidean spaces (σ₀, σ₁).
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import json
from pathlib import Path


@dataclass
class Demo1Config:
    """Configuration for Demo 1 multi-component inference experiments.

    This dataclass encapsulates all parameters needed to run the inference
    problem, organized into several categories:

    Model Space Parameters:
        N: Number of basis functions for each function component
        N_d: Number of data points
        N_p: Number of parameters to infer

    Prior Parameters - vp component:
        s_vp: Smoothness parameter for Bessel-Sobolev kernel
        length_scale_vp: Length scale for spatial correlations
        overall_variance_vp: Overall variance scaling

    Prior Parameters - vs component:
        s_vs: Smoothness parameter for Bessel-Sobolev kernel
        length_scale_vs: Length scale for spatial correlations
        overall_variance_vs: Overall variance scaling

    Prior Parameters - rho component:
        s_rho: Smoothness parameter for Bessel-Sobolev kernel
        length_scale_rho: Length scale for spatial correlations
        overall_variance_rho: Overall variance scaling

    Prior Parameters - sigma components:
        sigma_0_variance: Prior variance for σ₀
        sigma_1_variance: Prior variance for σ₁

    Numerical Parameters:
        integration_N: Number of integration points for operators
        kl_truncation_vp: Number of KL modes for vp prior
        kl_truncation_vs: Number of KL modes for vs prior
        kl_truncation_rho: Number of KL modes for rho prior

    Inference Parameters:
        noise_level: Standard deviation of observation noise
        compute_model_posterior: Whether to compute full posterior (slow) or just mean/variance
        parallel: Whether to use parallel computation
        n_jobs: Number of parallel jobs (-1 for all cores)

    Experiment Metadata:
        name: Descriptive name for this configuration
        description: Longer description of experiment purpose
    """

    # Model space parameters
    N: int = 100
    N_d: int = 50
    N_p: int = 20

    # Prior parameters - vp component
    s_vp: float = 2.0
    length_scale_vp: float = 0.3
    overall_variance_vp: float = 1.0

    # Prior parameters - vs component
    s_vs: float = 2.0
    length_scale_vs: float = 0.3
    overall_variance_vs: float = 1.0

    # Prior parameters - rho component
    s_rho: float = 2.0
    length_scale_rho: float = 0.3
    overall_variance_rho: float = 1.0

    # Prior parameters - sigma components
    sigma_0_variance: float = 1.0
    sigma_1_variance: float = 1.0

    # Numerical parameters
    integration_N: int = 200
    kl_truncation_vp: Optional[int] = None
    kl_truncation_vs: Optional[int] = None
    kl_truncation_rho: Optional[int] = None

    # Inference parameters
    noise_level: float = 0.01
    compute_model_posterior: bool = False
    parallel: bool = False
    n_jobs: int = -1

    # Experiment metadata
    name: str = "demo_1_default"
    description: str = "Default configuration for Demo 1 multi-component inference"

    # Random seed for reproducibility
    random_seed: Optional[int] = None

    def __post_init__(self):
        """Validate and derive additional parameters."""
        # Validate positive values
        if self.N <= 0:
            raise ValueError(f"N must be positive, got {self.N}")
        if self.N_d <= 0:
            raise ValueError(f"N_d must be positive, got {self.N_d}")
        if self.N_p <= 0:
            raise ValueError(f"N_p must be positive, got {self.N_p}")

        # Validate smoothness parameters (s > 0.5 for convergence)
        for s_name, s_val in [("s_vp", self.s_vp), ("s_vs", self.s_vs), ("s_rho", self.s_rho)]:
            if s_val <= 0.5:
                raise ValueError(f"{s_name} must be > 0.5 for convergence, got {s_val}")

        # Validate length scales
        for ls_name, ls_val in [("length_scale_vp", self.length_scale_vp),
                                 ("length_scale_vs", self.length_scale_vs),
                                 ("length_scale_rho", self.length_scale_rho)]:
            if ls_val <= 0:
                raise ValueError(f"{ls_name} must be positive, got {ls_val}")

        # Validate variances
        for var_name, var_val in [("overall_variance_vp", self.overall_variance_vp),
                                   ("overall_variance_vs", self.overall_variance_vs),
                                   ("overall_variance_rho", self.overall_variance_rho),
                                   ("sigma_0_variance", self.sigma_0_variance),
                                   ("sigma_1_variance", self.sigma_1_variance)]:
            if var_val <= 0:
                raise ValueError(f"{var_name} must be positive, got {var_val}")

        # Validate noise level
        if self.noise_level <= 0:
            raise ValueError(f"noise_level must be positive, got {self.noise_level}")

        # Validate KL truncations if specified
        for kl_name, kl_val in [("kl_truncation_vp", self.kl_truncation_vp),
                                 ("kl_truncation_vs", self.kl_truncation_vs),
                                 ("kl_truncation_rho", self.kl_truncation_rho)]:
            if kl_val is not None and kl_val <= 0:
                raise ValueError(f"{kl_name} must be positive if specified, got {kl_val}")

    @property
    def k_vp(self) -> float:
        """Compute k parameter from length scale for vp component."""
        return 1.0 / self.length_scale_vp

    @property
    def k_vs(self) -> float:
        """Compute k parameter from length scale for vs component."""
        return 1.0 / self.length_scale_vs

    @property
    def k_rho(self) -> float:
        """Compute k parameter from length scale for rho component."""
        return 1.0 / self.length_scale_rho

    @property
    def alpha_vp(self) -> float:
        """Compute alpha parameter from overall variance for vp component."""
        return self.overall_variance_vp * (2 * self.k_vp) ** (2 * self.s_vp)

    @property
    def alpha_vs(self) -> float:
        """Compute alpha parameter from overall variance for vs component."""
        return self.overall_variance_vs * (2 * self.k_vs) ** (2 * self.s_vs)

    @property
    def alpha_rho(self) -> float:
        """Compute alpha parameter from overall variance for rho component."""
        return self.overall_variance_rho * (2 * self.k_rho) ** (2 * self.s_rho)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        d = asdict(self)
        # Add derived parameters
        d['k_vp'] = self.k_vp
        d['k_vs'] = self.k_vs
        d['k_rho'] = self.k_rho
        d['alpha_vp'] = self.alpha_vp
        d['alpha_vs'] = self.alpha_vs
        d['alpha_rho'] = self.alpha_rho
        return d

    def save(self, path: Path) -> None:
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'Demo1Config':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)

        # Remove derived parameters if present
        for key in ['k_vp', 'k_vs', 'k_rho', 'alpha_vp', 'alpha_vs', 'alpha_rho']:
            data.pop(key, None)

        return cls(**data)

    def copy(self, **changes) -> 'Demo1Config':
        """Create a copy of this configuration with specified changes."""
        d = self.to_dict()
        # Remove derived parameters
        for key in ['k_vp', 'k_vs', 'k_rho', 'alpha_vp', 'alpha_vs', 'alpha_rho']:
            d.pop(key, None)
        d.update(changes)
        return Demo1Config(**d)


# Pre-defined configurations for common scenarios
def get_fast_config() -> Demo1Config:
    """Fast configuration for quick testing (reduced resolution)."""
    return Demo1Config(
        N=50,
        N_d=25,
        N_p=10,
        integration_N=100,
        kl_truncation_vp=20,
        kl_truncation_vs=20,
        kl_truncation_rho=20,
        compute_model_posterior=False,
        parallel=False,
        name="fast_test",
        description="Fast configuration for testing with reduced resolution"
    )


def get_standard_config() -> Demo1Config:
    """Standard configuration matching demo_1.ipynb defaults."""
    return Demo1Config(
        N=100,
        N_d=50,
        N_p=20,
        integration_N=200,
        kl_truncation_vp=None,
        kl_truncation_vs=None,
        kl_truncation_rho=None,
        compute_model_posterior=False,
        parallel=True,
        n_jobs=-1,
        name="standard",
        description="Standard configuration matching demo_1.ipynb"
    )


def get_high_resolution_config() -> Demo1Config:
    """High resolution configuration for detailed analysis."""
    return Demo1Config(
        N=200,
        N_d=100,
        N_p=40,
        integration_N=400,
        kl_truncation_vp=None,
        kl_truncation_vs=None,
        kl_truncation_rho=None,
        compute_model_posterior=False,
        parallel=True,
        n_jobs=-1,
        name="high_resolution",
        description="High resolution configuration for detailed analysis"
    )


def get_posterior_sampling_config() -> Demo1Config:
    """Configuration with full posterior computation enabled."""
    return Demo1Config(
        N=100,
        N_d=50,
        N_p=20,
        integration_N=200,
        kl_truncation_vp=50,
        kl_truncation_vs=50,
        kl_truncation_rho=50,
        compute_model_posterior=True,
        parallel=True,
        n_jobs=-1,
        name="posterior_sampling",
        description="Configuration with full posterior computation for sampling"
    )
