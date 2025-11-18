"""
Configuration objects for numerical integration and parallelization parameters.

This module provides a clean way to manage integration and parallel computation
parameters across different subsystems without polluting __init__ signatures.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class IntegrationConfig:
    """
    Configuration for numerical integration parameters.

    This dataclass provides a clean, type-safe way to manage integration
    settings that may differ across various subsystems (e.g., inner products,
    dual space operations, operator applications).

    Attributes:
        method: Integration method ('simpson', 'trapz', 'quad')
        n_points: Number of quadrature points for simpson/trapz

    Example:
        >>> # Use defaults
        >>> config = IntegrationConfig()
        >>>
        >>> # Override specific parameters
        >>> config = IntegrationConfig(n_points=10000)
        >>>
        >>> # Modify after creation
        >>> config.n_points = 5000
    """

    method: Literal['simpson', 'trapz', 'quad'] = 'simpson'
    n_points: int = 1000

    def copy(self, **overrides) -> 'IntegrationConfig':
        """
        Create a copy with optional parameter overrides.

        Example:
            >>> base = IntegrationConfig(n_points=1000)
            >>> high_res = base.copy(n_points=10000)
        """
        import copy
        new_config = copy.copy(self)
        for key, value in overrides.items():
            if hasattr(new_config, key):
                setattr(new_config, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
        return new_config

    @classmethod
    def high_accuracy(cls) -> 'IntegrationConfig':
        """Preset for high-accuracy integration."""
        return cls(method='simpson', n_points=10000)

    @classmethod
    def fast(cls) -> 'IntegrationConfig':
        """Preset for fast, lower-accuracy integration."""
        return cls(method='trapz', n_points=500)

    @classmethod
    def adaptive(cls, dim: int) -> 'IntegrationConfig':
        """
        Create config with points scaled to basis dimension.

        For spectral methods, higher mode numbers require more integration
        points to maintain accuracy.

        Args:
            dim: Dimension of the basis (number of modes)

        Returns:
            Config with n_points = max(1000, 100 * dim)
        """
        n_points = max(1000, 100 * dim)
        return cls(n_points=n_points)


@dataclass
class ParallelConfig:
    """
    Configuration for parallel computation parameters.

    This dataclass manages parallelization settings separately from
    integration configuration, allowing fine-grained control over
    when and how parallel computation is used.

    Attributes:
        enabled: Whether to use parallel computation
        n_jobs: Number of parallel jobs (-1 = all cores, 1 = no parallel)

    Example:
        >>> # Use defaults (no parallelization)
        >>> config = ParallelConfig()
        >>>
        >>> # Enable parallelization with all cores
        >>> config = ParallelConfig(enabled=True, n_jobs=-1)
        >>>
        >>> # Use specific number of cores
        >>> config = ParallelConfig(enabled=True, n_jobs=4)
    """

    enabled: bool = False
    n_jobs: int = -1

    def copy(self, **overrides) -> 'ParallelConfig':
        """
        Create a copy with optional parameter overrides.

        Example:
            >>> base = ParallelConfig(enabled=False)
            >>> parallel = base.copy(enabled=True, n_jobs=4)
        """
        import copy
        new_config = copy.copy(self)
        for key, value in overrides.items():
            if hasattr(new_config, key):
                setattr(new_config, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
        return new_config

    @classmethod
    def all_cores(cls) -> 'ParallelConfig':
        """Preset for parallel computation using all available cores."""
        return cls(enabled=True, n_jobs=-1)

    @classmethod
    def cores(cls, n: int) -> 'ParallelConfig':
        """
        Preset for parallel computation using specific number of cores.

        Args:
            n: Number of cores to use
        """
        return cls(enabled=True, n_jobs=n)

    @classmethod
    def serial(cls) -> 'ParallelConfig':
        """Preset for serial (non-parallel) computation."""
        return cls(enabled=False, n_jobs=1)


@dataclass
class LebesgueIntegrationConfig:
    """
    Hierarchical integration configuration for Lebesgue spaces.

    This allows different integration settings for:
    - Inner products (used by l2_inner_product, Gram matrices)
    - Dual operations (used by LinearFormKernel, to_dual/from_dual)
    - General operations (fallback for other integrations)

    Example:
        >>> config = LebesgueIntegrationConfig()
        >>>
        >>> # High accuracy for inner products (Gram matrices)
        >>> config.inner_product.n_points = 10000
        >>>
        >>> # Keep default for general operations
        >>> # config.general uses default 1000 points
    """

    inner_product: IntegrationConfig = field(default_factory=IntegrationConfig)
    dual: IntegrationConfig = field(default_factory=IntegrationConfig)
    general: IntegrationConfig = field(default_factory=IntegrationConfig)

    @classmethod
    def from_single(cls, config: IntegrationConfig) -> 'LebesgueIntegrationConfig':
        """Create hierarchical config using same settings for all subsystems."""
        return cls(
            inner_product=config.copy(),
            dual=config.copy(),
            general=config.copy()
        )

    @classmethod
    def high_accuracy_galerkin(cls) -> 'LebesgueIntegrationConfig':
        """
        Preset optimized for accurate Galerkin matrix assembly.

        Uses high-accuracy integration for inner products (which are used
        in Galerkin matrices) and dual space operations.
        """
        return cls(
            inner_product=IntegrationConfig(method='simpson', n_points=10000),
            dual=IntegrationConfig(method='simpson', n_points=10000),
            general=IntegrationConfig()  # default for less critical ops
        )

    @classmethod
    def adaptive_spectral(cls, dim: int) -> 'LebesgueIntegrationConfig':
        """
        Preset for spectral methods with adaptive point selection.

        Scales integration points with basis dimension to maintain accuracy
        for high-frequency modes.

        Args:
            dim: Number of basis functions
        """
        n_points = max(1000, 100 * dim)
        return cls(
            inner_product=IntegrationConfig(n_points=n_points),
            dual=IntegrationConfig(n_points=n_points),
            general=IntegrationConfig(n_points=n_points)
        )


@dataclass
class LebesgueParallelConfig:
    """
    Hierarchical parallelization configuration for Lebesgue spaces.

    This allows different parallel settings for:
    - Inner products (used by l2_inner_product, Gram matrices)
    - Dual operations (used by LinearFormKernel, to_dual/from_dual)
    - General operations (fallback for other operations)

    Example:
        >>> config = LebesgueParallelConfig()
        >>>
        >>> # Enable parallelization for dual operations
        >>> config.dual.enabled = True
        >>> config.dual.n_jobs = 8
        >>>
        >>> # Keep serial for inner products
        >>> # config.inner_product.enabled = False (default)
    """

    inner_product: ParallelConfig = field(default_factory=ParallelConfig)
    dual: ParallelConfig = field(default_factory=ParallelConfig)
    general: ParallelConfig = field(default_factory=ParallelConfig)

    @classmethod
    def from_single(cls, config: ParallelConfig) -> 'LebesgueParallelConfig':
        """Create hierarchical config using same settings for all subsystems."""
        return cls(
            inner_product=config.copy(),
            dual=config.copy(),
            general=config.copy()
        )

    @classmethod
    def parallel_dual(cls, n_jobs: int = -1) -> 'LebesgueParallelConfig':
        """
        Preset with parallelization enabled only for dual operations.

        This is often the most useful pattern: dual operations (LinearFormKernel)
        can benefit from parallelization while inner products are fast enough
        serially.

        Args:
            n_jobs: Number of cores for dual operations (-1 = all cores)
        """
        return cls(
            inner_product=ParallelConfig.serial(),
            dual=ParallelConfig(enabled=True, n_jobs=n_jobs),
            general=ParallelConfig.serial()
        )

    @classmethod
    def all_parallel(cls, n_jobs: int = -1) -> 'LebesgueParallelConfig':
        """
        Preset with parallelization enabled for all operations.

        Args:
            n_jobs: Number of cores (-1 = all cores)
        """
        parallel = ParallelConfig(enabled=True, n_jobs=n_jobs)
        return cls.from_single(parallel)

    @classmethod
    def serial(cls) -> 'LebesgueParallelConfig':
        """Preset with serial (non-parallel) computation for all operations."""
        return cls.from_single(ParallelConfig.serial())
