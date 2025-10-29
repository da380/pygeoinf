# Bessel-Sobolev Parameter Sweep

## Overview

Added a comprehensive parameter sweep for exploring the Bessel-Sobolev prior parameters `k` and `s` in the PLI inference framework.

## New Sweep Function

**Function:** `sweep_bessel_sobolev_params()` in `example_sweep.py`

**Purpose:** Systematically explore how the Bessel-Sobolev prior parameters affect inference quality and computational cost.

## Parameters Being Swept

### k: Bessel Parameter (Correlation Length Scale)
- **Values tested:** [0.5, 1.0, 2.0, 5.0]
- **Effect:**
  - Lower k → longer correlation length, smoother spatial variations
  - Higher k → shorter correlation length, more localized features
  - Controls the balance between identity (I) and Laplacian (L) in C₀ = (k²I + L)^(-s)

### s: Sobolev Order (Smoothness)
- **Values tested:** [0.5, 1.0, 1.5, 2.0]
- **Effect:**
  - Lower s → less smooth prior samples, more flexible
  - Higher s → smoother prior samples, stronger regularization
  - s = 1 is typical, s = 2 gives very smooth priors
  - Controls regularity of the covariance operator

## Total Experiments

**4 k values × 4 s values = 16 experiments**

Each experiment tests a different combination of correlation length and smoothness.

## Configuration

```python
base_config = {
    'N': 100,           # Model space dimension
    'N_d': 30,          # Number of data points
    'N_p': 20,          # Number of properties
    'K': 100,           # KL expansion modes
    'basis': None,      # Use default basis
    'noise_level': 0.1, # 10% noise
    'compute_model_posterior': False,
    'random_seed': 42,
    'n_jobs': 30,       # Parallel workers

    # Bessel-Sobolev prior
    'prior_type': 'bessel_sobolev',
    'alpha': 0.1,       # Laplacian scaling
    'method': 'spectral',
    'dofs': 100,
    'n_samples': 1024,  # Fast transform samples
    'use_fast_transforms': True,
    'bc_config': {'bc_type': 'dirichlet', 'left': 0, 'right': 0}
}
```

## Visualizations Generated

The sweep creates a comprehensive analysis plot with 9 panels:

### 1. Heatmaps (3 panels)
- **Property RMS error** vs k and s
- **Model RMS error** vs k and s
- **Inference time** vs k and s

### 2. Line Plots (4 panels)
- Property error vs k (for different s values)
- Property error vs s (for different k values)
- Scatter: Property error vs Model error (colored by k, sized by s)
- Inference time vs k (for different s values)

### 3. Statistical Analysis (2 panels)
- **Uncertainty calibration:** % of properties within 2σ
- **Optimal parameters:** Best accuracy, fastest, best calibration

## Output Files

In `experiments/bessel_sobolev_params_YYYYMMDD_HHMMSS/`:

1. **detailed_results.csv** - All experiment results
2. **summary_statistics.csv** - Grouped statistics by k and s
3. **analysis_bessel_sobolev.png** - Comprehensive visualization (9 panels)
4. **Individual experiment directories** (16 total):
   - `k0.5_s0.5/`, `k0.5_s1.0/`, etc.
   - Each contains: config.json, timings.json, metrics.json, figures/

## Usage

### Run the Sweep
```bash
cd pygeoinf/interval/demos
python example_sweep.py 6
```

### Run from Python
```python
from example_sweep import sweep_bessel_sobolev_params
sweep_dir = sweep_bessel_sobolev_params()
```

### Analyze Results
```python
import pandas as pd
df = pd.read_csv('experiments/bessel_sobolev_params_*/detailed_results.csv')

# Find optimal parameters
best_accuracy = df.loc[df['property_rms_error'].idxmin()]
print(f"Best accuracy: k={best_accuracy['k']}, s={best_accuracy['s']}")

# Compare different s values for fixed k
k1_results = df[df['k'] == 1.0]
print(k1_results[['s', 'property_rms_error', 'inference_total']])
```

## Expected Results

### Trends to Observe

1. **Accuracy vs k:**
   - Too small k → over-smooth, poor data fit
   - Too large k → under-regularized, noisy reconstruction
   - Optimal k balances smoothness and data fidelity

2. **Accuracy vs s:**
   - Higher s → smoother solutions, may miss sharp features
   - Lower s → more flexible, better fit but potentially noisier
   - s = 1.0 often provides good balance

3. **Computational Cost:**
   - Should be relatively constant across k and s
   - Fast spectral transforms make this efficient
   - All experiments ~2-4 seconds each

4. **Uncertainty Calibration:**
   - Well-calibrated priors: ~95% properties within 2σ
   - Over-regularized: too many within 2σ (overconfident)
   - Under-regularized: too few within 2σ (underconfident)

## Scientific Questions Addressed

1. **How does correlation length (k) affect inference quality?**
   - Measures sensitivity to spatial scale of prior assumptions
   - Identifies optimal correlation length for given data spacing

2. **How does smoothness order (s) affect reconstruction?**
   - Quantifies trade-off between smoothness and flexibility
   - Determines appropriate regularization strength

3. **Are there optimal (k, s) combinations?**
   - Identifies parameter ranges that work well together
   - May reveal sweet spots for this problem class

4. **How do k and s interact?**
   - Understanding coupled effects
   - Non-linear relationships between parameters and accuracy

## Integration with Other Sweeps

This sweep complements:
- **Example 1:** KL expansion sweep → find optimal K for given k, s
- **Example 4:** Boundary condition sweep → compare k, s effects across BCs
- **Example 5:** Data/BC/method sweep → understand k, s in different contexts

## Customization

To test different parameter ranges:

```python
# More values for finer resolution
param_grid = {
    'k': [0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
    's': [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
}

# Logarithmic spacing for k
import numpy as np
param_grid = {
    'k': np.logspace(-1, 1, 5),  # 0.1 to 10
    's': np.linspace(0.5, 2.5, 5)
}
```

## Mathematical Background

### Bessel-Sobolev Prior
The prior covariance operator is:

$$C_0 = (k^2 I + L)^{-s}$$

where:
- $L = -\alpha \nabla^2$ is the negative Laplacian
- $k^2$ is the Bessel parameter
- $s$ is the Sobolev order

### Spectral Representation
For eigenfunctions $\phi_i$ with eigenvalues $\lambda_i$ of $L$:

$$C_0 = \sum_{i=0}^{\infty} \frac{1}{(k^2 + \lambda_i)^s} \phi_i \otimes \phi_i$$

### Parameter Effects

**Correlation Length:**
$$\ell \sim k^{-1}$$

Smaller k → larger correlation length → smoother variations

**Effective Eigenvalue Decay:**
$$\mu_i = (k^2 + \lambda_i)^{-s}$$

- Small $\lambda_i$ (smooth modes): dominated by $k^2$ term
- Large $\lambda_i$ (rough modes): Laplacian dominates, strong decay

## Date
October 29, 2024

## Related Files
- `example_sweep.py` - Contains `sweep_bessel_sobolev_params()`
- `run_pli_experiments.py` - PLIExperiment class with Bessel-Sobolev support
- `BESSEL_SOBOLEV_PRIOR_UPDATE.md` - Documentation of prior implementation
