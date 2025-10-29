# PLI Multiple Notebook - Bessel-Sobolev Prior Update

## Summary

Updated `pli_multiple.ipynb` to use the **Bessel-Sobolev inverse operator** as the prior covariance for both the $v_p$ and $v_s$ components in the multi-parameter inverse problem.

## Changes Made

### 1. Updated Imports
Added `Laplacian` and `BesselSobolevInverse` to the imports:
```python
from pygeoinf.interval.operators import InverseLaplacian, Laplacian, BesselSobolevInverse
```

### 2. Modified Prior Covariance Setup

**Before:**
```python
# Simple inverse Laplacian
C_0_vp = InverseLaplacian(M_vp, bc_dirichlet, alpha_vp, method='fem', dofs=100)
C_0_vs = InverseLaplacian(M_vs, bc_dirichlet, alpha_vs, method='fem', dofs=100)
```

**After:**
```python
# Bessel-Sobolev parameters
k_vp = 1.0      # Correlation length scale
s_vp = 1.0      # Smoothness order
alpha_vp = 0.1  # Laplacian scaling

k_vs = 1.0
s_vs = 1.0
alpha_vs = 0.05

# Create Laplacian operators
L_vp = Laplacian(M_vp, bc_dirichlet, alpha_vp, method='spectral',
                 dofs=100, n_samples=1024)
L_vs = Laplacian(M_vs, bc_dirichlet, alpha_vs, method='spectral',
                 dofs=100, n_samples=1024)

# Create Bessel-Sobolev inverse operators
C_0_vp = BesselSobolevInverse(M_vp, M_vp, k_vp, s_vp, L_vp,
                               dofs=100, n_samples=1024, use_fast_transforms=True)
C_0_vs = BesselSobolevInverse(M_vs, M_vs, k_vs, s_vs, L_vs,
                               dofs=100, n_samples=1024, use_fast_transforms=True)
```

### 3. Updated Documentation

Added comprehensive markdown cells explaining:
- The Bessel-Sobolev prior formulation: $C_0 = (k^2 I + L)^{-s}$
- Parameter meanings and effects
- Benefits over simple inverse Laplacian
- Parameter selection guidance

### 4. Enhanced Diagnostics

Added output showing:
- Bessel-Sobolev parameter values
- Fast transform usage
- Eigenvalue scaling formula
- First few eigenvalues of both $L$ and $C_0$

## New Prior Formulation

### Mathematical Form
$$C_0 = (k^2 I + L)^{-s}$$

where:
- $L = -\alpha \nabla^2$ is the negative Laplacian
- $k$ controls correlation length: $\ell \sim k^{-1}$
- $s$ controls smoothness/regularity

### Spectral Representation
For eigenfunctions $\phi_i$ with eigenvalues $\lambda_i$ of $L$:

$$C_0 = \sum_{i=0}^{\infty} \frac{1}{(k^2 + \lambda_i)^s} \phi_i \otimes \phi_i$$

### Current Configuration

**$v_p$ component:**
- $k_{vp} = 1.0$ (moderate correlation length)
- $s_{vp} = 1.0$ (standard smoothness)
- $\alpha_{vp} = 0.1$ (Laplacian scaling)

**$v_s$ component:**
- $k_{vs} = 1.0$ (moderate correlation length)
- $s_{vs} = 1.0$ (standard smoothness)
- $\alpha_{vs} = 0.05$ (weaker Laplacian scaling → smoother)

## Benefits

1. **More Flexible**: Independent control of correlation length (k) and smoothness (s)
2. **Better Conditioned**: Well-behaved for high smoothness orders
3. **Fast Computation**: Leverages fast DCT/DST transforms (recently fixed)
4. **Tunable**: Can adjust parameters per component (vp vs vs)
5. **Standard**: Widely used in Bayesian inverse problems

## Parameter Tuning

To find optimal parameters, you can:

1. **Run parameter sweep**:
   ```bash
   cd pygeoinf/interval/demos
   python example_sweep.py 6
   ```
   This tests 16 combinations of k and s values.

2. **Manual experimentation**:
   - Adjust `k_vp`, `k_vs`: Try [0.5, 1.0, 2.0, 5.0]
   - Adjust `s_vp`, `s_vs`: Try [0.5, 1.0, 1.5, 2.0]
   - Compare reconstruction quality and uncertainty calibration

3. **Physical intuition**:
   - If true model has sharp features: use smaller s, larger k
   - If true model is smooth: use larger s, smaller k
   - Match correlation length k to data spacing

## Backward Compatibility

To revert to simple inverse Laplacian (legacy behavior):
```python
# Use inverse Laplacian instead
C_0_vp = InverseLaplacian(M_vp, bc_dirichlet, alpha_vp, method='fem', dofs=100)
C_0_vs = InverseLaplacian(M_vs, bc_dirichlet, alpha_vs, method='fem', dofs=100)
```

Note: The inverse Laplacian is approximately equivalent to Bessel-Sobolev with $k^2 \to 0$ and $s = 1$.

## Performance

- **No slowdown**: Fast spectral transforms make this efficient
- **Memory**: Same as inverse Laplacian
- **Convergence**: May improve with better-conditioned operators

## Testing

The notebook should run without modifications to:
- Forward operator setup
- Data generation
- Inference workflow
- Visualization code

Only the prior construction changes.

## Related Files

- `run_pli_experiments.py`: Single-parameter version with Bessel-Sobolev support
- `example_sweep.py`: Contains `sweep_bessel_sobolev_params()` for parameter exploration
- `BESSEL_SOBOLEV_PRIOR_UPDATE.md`: Detailed documentation of the operator
- `BESSEL_SOBOLEV_SWEEP.md`: Parameter sweep documentation

## Date
October 29, 2024
