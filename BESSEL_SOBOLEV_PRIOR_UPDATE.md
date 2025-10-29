# Bessel-Sobolev Prior Model Covariance Update

## Summary

Modified `run_pli_experiments.py` to use the **Bessel-Sobolev inverse operator** (`BesselSobolevInverse`) as the default prior model covariance instead of the simple inverse Laplacian.

## Changes Made

### 1. Imports
Added imports for `Laplacian` and `BesselSobolevInverse`:
```python
from pygeoinf.interval.operators import (
    InverseLaplacian, Laplacian, BesselSobolevInverse
)
```

### 2. New Configuration Parameters

#### Prior Type Selection
- `prior_type` (default: `'bessel_sobolev'`): Selects the prior covariance operator
  - `'bessel_sobolev'`: C₀ = (k²I + L)^(-s)
  - `'inverse_laplacian'`: C₀ = (αL)^(-1)

#### Bessel-Sobolev Parameters
When `prior_type='bessel_sobolev'` (default):
- `k` (default: 1.0): Bessel parameter k²
- `s` (default: 1.0): Sobolev order s
- `alpha` (default: 0.1): Laplacian scaling parameter
- `method` (default: 'spectral'): Discretization method ('spectral' or 'fd')
- `dofs` (default: 100): Degrees of freedom
- `n_samples` (default: 1024): Number of samples for fast spectral transforms
- `use_fast_transforms` (default: True): Enable fast DCT/DST transforms

#### Inverse Laplacian Parameters
When `prior_type='inverse_laplacian'`:
- `alpha`: Prior regularization parameter
- `method`: Discretization method
- `dofs`: Degrees of freedom

### 3. Updated Prior Setup (`_setup_prior` method)

The prior covariance operator is now created based on `prior_type`:

```python
if self.prior_type == 'inverse_laplacian':
    # C_0 = (αL)^(-1) - simple inverse Laplacian
    C_0 = InverseLaplacian(
        self.M, bc, self.alpha,
        method=self.method, dofs=self.dofs
    )
elif self.prior_type == 'bessel_sobolev':
    # C_0 = (k²I + L)^(-s) - Bessel-Sobolev inverse
    # First create the Laplacian operator L
    L = Laplacian(
        self.M, bc, self.alpha,
        method=self.method, dofs=self.dofs,
        n_samples=self.n_samples
    )
    # Then create the Bessel-Sobolev inverse
    C_0 = BesselSobolevInverse(
        self.M, self.M, self.k, self.s, L,
        dofs=self.dofs, n_samples=self.n_samples,
        use_fast_transforms=self.use_fast_transforms
    )
```

### 4. Updated Documentation

The module docstring now includes comprehensive documentation of:
- Prior covariance operator types
- Bessel-Sobolev parameters (k, s, alpha, method, dofs, n_samples, use_fast_transforms)
- Inverse Laplacian parameters (alpha, method, dofs)

### 5. Updated Example Configurations

Both example functions now use the Bessel-Sobolev prior by default:

```python
base_config = {
    'N': 100,
    'N_d': 50,
    'N_p': 20,
    'basis': 'sine',
    'noise_level': 0.1,
    'compute_model_posterior': False,
    'random_seed': 42,
    # Bessel-Sobolev prior parameters (default)
    'prior_type': 'bessel_sobolev',
    'k': 1.0,
    's': 1.0,
    'alpha': 0.1,
    'method': 'spectral',
    'dofs': 100,
    'n_samples': 1024,
    'use_fast_transforms': True
}
```

## Mathematical Background

### Bessel-Sobolev Operator
The Bessel-Sobolev operator is defined as:
```
C₀ = (k²I + L)^(-s)
```

where:
- L is the negative Laplacian operator: L = -α∇²
- k² is the Bessel parameter (controls the correlation length scale)
- s is the Sobolev order (controls smoothness)
- Higher s → smoother realizations
- Higher k → shorter correlation length

### Comparison to Inverse Laplacian
The inverse Laplacian is a special case:
```
C₀ = (αL)^(-1) ≈ (k²I + L)^(-s) when k²→0, s→1
```

The Bessel-Sobolev operator provides:
- More flexible control over smoothness (via s)
- Explicit correlation length scale (via k)
- Well-conditioned operators even for high smoothness
- Fast spectral transform methods for efficiency

## Usage Examples

### Default: Bessel-Sobolev Prior
```python
config = {
    'N': 100,
    'N_d': 50,
    'N_p': 20,
    'K': 100,
    'basis': None,
    'noise_level': 0.1,
    # Bessel-Sobolev (default - no need to specify prior_type)
    'k': 1.0,      # Correlation length scale
    's': 1.0,      # Smoothness order
    'alpha': 0.1,  # Laplacian scaling
}
```

### Using Inverse Laplacian (legacy)
```python
config = {
    'N': 100,
    'N_d': 50,
    'N_p': 20,
    'K': 100,
    'basis': None,
    'noise_level': 0.1,
    'prior_type': 'inverse_laplacian',  # Explicitly select
    'alpha': 0.1,
    'method': 'spectral',
    'dofs': 100
}
```

### Sweeping Over Bessel-Sobolev Parameters
```python
param_grid = {
    'k': [0.5, 1.0, 2.0],    # Correlation length
    's': [0.5, 1.0, 1.5],    # Smoothness
    'K': [50, 100, 150]      # KL modes
}
```

## Benefits

1. **More Flexible Priors**: Can control smoothness and correlation length independently
2. **Better Conditioning**: Bessel-Sobolev operators are well-conditioned even for high smoothness
3. **Fast Computation**: Leverages fast DCT/DST transforms (fixed in recent bugfix)
4. **Backward Compatible**: Can still use inverse Laplacian by setting `prior_type='inverse_laplacian'`
5. **Standard in Literature**: Bessel-Sobolev priors are widely used in inverse problems

## Testing

The modifications maintain backward compatibility. To test:

1. **Bessel-Sobolev prior** (default):
   ```python
   from run_pli_experiments import PLIExperiment
   config = {'N': 100, 'N_d': 50, 'N_p': 20, 'K': 100,
             'basis': None, 'noise_level': 0.1}
   exp = PLIExperiment(config, output_dir)
   exp.run()
   ```

2. **Inverse Laplacian** (legacy):
   ```python
   config = {... 'prior_type': 'inverse_laplacian', 'alpha': 0.1}
   exp = PLIExperiment(config, output_dir)
   exp.run()
   ```

## Related Files

- `pygeoinf/interval/operators.py`: Contains `BesselSobolevInverse` and `Laplacian` classes
- `pygeoinf/interval/demos/run_pli_experiments.py`: Main file modified
- `pygeoinf/interval/demos/example_sweep.py`: Parameter sweep examples
- `BUGFIX_SUMMARY.md`: Recent fix for fast spectral transforms

## Date
October 29, 2024
