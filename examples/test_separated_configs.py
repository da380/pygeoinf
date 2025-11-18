"""
Test examples for separated integration and parallel configs.

This demonstrates the new API with separated concerns.
"""

from pygeoinf.interval import (
    Lebesgue,
    IntervalDomain,
    IntegrationConfig,
    LebesgueIntegrationConfig,
    ParallelConfig,
    LebesgueParallelConfig
)

# Create domain
domain = IntervalDomain(0, 1)

# ==============================================================================
# Example 1: Simple - just specify what you need
# ==============================================================================

print("Example 1: Simple modifications\n" + "="*50)

# Start with defaults
space = Lebesgue(100, domain, basis='sine')

# Change only accuracy (integration)
space.integration.inner_product.n_points = 20000
print(f"Integration points: {space.integration.inner_product.n_points}")

# Change only performance (parallelization) - independent!
space.parallel.dual.enabled = True
space.parallel.dual.n_jobs = 8
print(f"Parallel enabled: {space.parallel.dual.enabled}")
print(f"Number of jobs: {space.parallel.dual.n_jobs}")

# ==============================================================================
# Example 2: Use preset configurations
# ==============================================================================

print("\n\nExample 2: Preset configurations\n" + "="*50)

# Preset integration config (high accuracy)
int_cfg = LebesgueIntegrationConfig.high_accuracy_galerkin()
print(f"Preset integration: {int_cfg.inner_product.n_points} points")

# Preset parallel config (parallel dual operations)
par_cfg = LebesgueParallelConfig.parallel_dual(n_jobs=8)
print(f"Preset parallel: dual.enabled = {par_cfg.dual.enabled}")

# Create space with both presets
space2 = Lebesgue(
    100,
    domain,
    basis='sine',
    integration_config=int_cfg,
    parallel_config=par_cfg
)

print(f"Space2 inner_product points: {space2.integration.inner_product.n_points}")
print(f"Space2 dual parallel: {space2.parallel.dual.enabled}")

# ==============================================================================
# Example 3: Adaptive integration with environment-specific parallelization
# ==============================================================================

print("\n\nExample 3: Adaptive + environment-specific\n" + "="*50)

dim = 100

# Adaptive integration (scales with dimension)
int_cfg = LebesgueIntegrationConfig.adaptive_spectral(dim=dim)
print(f"Adaptive integration: {int_cfg.inner_product.n_points} points for dim={dim}")

# Environment-specific parallelization
import os
on_hpc = os.getenv('SLURM_JOB_ID') is not None  # Check if on HPC

if on_hpc:
    par_cfg = LebesgueParallelConfig.all_parallel(n_jobs=-1)
    print("Environment: HPC cluster - using all cores")
else:
    par_cfg = LebesgueParallelConfig.parallel_dual(n_jobs=4)
    print("Environment: laptop - using 4 cores for dual ops only")

space3 = Lebesgue(
    dim,
    domain,
    basis='sine',
    integration_config=int_cfg,
    parallel_config=par_cfg
)

# ==============================================================================
# Example 4: Individual ParallelConfig
# ==============================================================================

print("\n\nExample 4: Individual ParallelConfig\n" + "="*50)

# Create simple parallel config
par = ParallelConfig.all_cores()
print(f"Simple parallel config: enabled={par.enabled}, n_jobs={par.n_jobs}")

par2 = ParallelConfig.cores(4)
print(f"4-core config: enabled={par2.enabled}, n_jobs={par2.n_jobs}")

par3 = ParallelConfig.serial()
print(f"Serial config: enabled={par3.enabled}, n_jobs={par3.n_jobs}")

# ==============================================================================
# Example 5: Backward compatibility
# ==============================================================================

print("\n\nExample 5: Backward compatibility\n" + "="*50)

# Old API still works
space_old = Lebesgue(100, domain, basis='sine', integration_npoints=5000)
print(f"Old API: {space_old.integration_npoints} points")

# Old properties still work
space_old.integration_npoints = 10000
print(f"After modification: {space_old.integration_npoints} points")

# ==============================================================================
# Example 6: Access patterns
# ==============================================================================

print("\n\nExample 6: Access patterns\n" + "="*50)

space = Lebesgue(100, domain, basis='sine')

# Read current configs
print("\nCurrent integration configs:")
print(f"  inner_product: {space.integration.inner_product.n_points} points")
print(f"  dual: {space.integration.dual.n_points} points")
print(f"  general: {space.integration.general.n_points} points")

print("\nCurrent parallel configs:")
print(f"  inner_product: enabled={space.parallel.inner_product.enabled}")
print(f"  dual: enabled={space.parallel.dual.enabled}")
print(f"  general: enabled={space.parallel.general.enabled}")

# Modify independently
print("\nModifying configs independently...")
space.integration.inner_product.n_points = 15000
space.integration.dual.method = 'trapz'
space.parallel.dual.enabled = True
space.parallel.dual.n_jobs = 4

print(f"  inner_product: {space.integration.inner_product.n_points} points")
print(f"  dual: {space.integration.dual.method} method")
print(f"  dual parallel: enabled={space.parallel.dual.enabled}, n_jobs={space.parallel.dual.n_jobs}")

print("\n" + "="*50)
print("✅ All examples completed successfully!")
print("Integration and parallelization are now cleanly separated.")
