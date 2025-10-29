# PLI Automated Experiments - Quick Start Guide

## Installation

No additional installation needed - uses existing pygeoinf dependencies.

## Run Your First Sweep

```bash
cd /home/adrian/PhD/Inferences/pygeoinf/pygeoinf/interval/demos
python example_sweep.py 1
```

This will:
1. Run PLI inference with K = [20, 30, 50, 100, 150, 200]
2. Save all results to `experiments/sweep_YYYYMMDD_HHMMSS/`
3. Generate comparison plots
4. Create a CSV summary of all runs

## What Gets Created

```
experiments/sweep_20251028_143025/
├── sweep_config.json              # What parameters were tested
├── summary.csv                     # All results in one table
├── analysis_K_sweep.png           # Comparison plots
│
├── run_001_K20/
│   ├── config.json                # This run's parameters
│   ├── timings.json               # How long each step took
│   ├── metrics.json               # Accuracy metrics
│   └── figures/                   # All visualizations
│       ├── sensitivity_kernels.png
│       ├── target_kernels.png
│       ├── true_model.png
│       ├── synthetic_observations.png
│       ├── prior_measure_on_model_space.png
│       ├── property_prior_distribution.png
│       ├── model_posterior_mean.png
│       └── property_inference_results.png
│
├── run_002_K30/
│   └── ... (same structure)
│
... (one folder per parameter combination)
```

## Custom Sweep

Create a Python script:

```python
from run_pli_experiments import PLISweep

sweep = PLISweep(base_dir="my_experiments")

# What stays constant
base_config = {
    'N': 100,           # Model space dimension
    'N_d': 50,          # Data points
    'N_p': 20,          # Properties to infer
    'basis': 'sine',    # Basis functions
    'alpha': 0.1,       # Prior regularization
    'noise_level': 0.1, # 10% noise
    'compute_model_posterior': False,  # Fast workflow
    'random_seed': 42
}

# What varies
param_grid = {
    'K': [50, 100, 200],              # Try 3 values
    'noise_level': [0.05, 0.1, 0.2]   # And 3 noise levels
}
# This creates 3 × 3 = 9 experiments

# Run it!
results = sweep.run(param_grid, base_config)

# Analyze
print(results[['K', 'noise_level', 'property_rms_error']])
```

## Predefined Examples

```bash
# Example 1: Find optimal K
python example_sweep.py 1

# Example 2: Test different resolutions
python example_sweep.py 2

# Example 3: Noise sensitivity
python example_sweep.py 3

# Run all examples
python example_sweep.py 4
```

## Analyzing Results

### Load Results from Disk

```python
import pandas as pd

# Load the summary CSV
df = pd.read_csv('experiments/sweep_20251028_143025/summary.csv')

# Filter successful runs
df_good = df[df['status'] == 'success']

# Find best configuration
best_idx = df_good['property_rms_error'].idxmin()
best_config = df_good.loc[best_idx]
print(f"Best config: K={best_config['K']}, error={best_config['property_rms_error']}")

# Compare timings
print(df_good[['K', 'inference_total']].sort_values('K'))
```

### Plot Custom Comparisons

```python
import matplotlib.pyplot as plt

# Plot speed vs accuracy tradeoff
plt.figure(figsize=(10, 6))
plt.scatter(df_good['inference_total'], df_good['property_rms_error'],
           s=100, alpha=0.6)

for idx, row in df_good.iterrows():
    plt.annotate(f"K={row['K']}",
                (row['inference_total'], row['property_rms_error']))

plt.xlabel('Time (s)')
plt.ylabel('RMS Error')
plt.title('Speed vs Accuracy Tradeoff')
plt.grid(True, alpha=0.3)
plt.savefig('tradeoff.png')
```

## Available Metrics

Every run computes:

**Accuracy:**
- `property_rms_error` - How accurate are the properties?
- `model_rms_error` - How accurate is the model?
- `properties_within_2sigma_pct` - How good is uncertainty quantification?

**Performance:**
- `inference_total` - Total inference time
- `model_posterior_compute` - Time for model posterior
- `property_covariance_extract` - Time for property covariance

**Data Fit:**
- `data_misfit_posterior` - How well does posterior fit data?
- `data_fit_improvement_pct` - Improvement over prior

See `EXPERIMENTS_README.md` for complete list.

## Tips

1. **Start small**: Test one parameter with 3-4 values first
2. **Use fast mode**: Keep `compute_model_posterior=False`
3. **Check failures**: If a run fails, check `summary.csv` for error messages
4. **Compare visually**: Open the `figures/` folders side-by-side
5. **Track experiments**: Each sweep gets a timestamp - keep notes on what you tested

## Common Use Cases

### Find optimal K for your problem
```python
param_grid = {'K': [20, 50, 100, 150, 200, 300, 500]}
```

### Test if more data helps
```python
param_grid = {'N_d': [10, 25, 50, 100, 200]}
```

### Compare basis functions
```python
param_grid = {'basis': ['sine', 'cosine', 'fourier']}
```

### Study impact of noise
```python
param_grid = {
    'noise_level': [0.01, 0.05, 0.1, 0.2],
    'K': [50, 100]
}
```

## Getting Help

- Read `EXPERIMENTS_README.md` for detailed documentation
- Look at `run_pli_experiments.py` for implementation
- Check `example_sweep.py` for working examples
- Each run saves `config.json` - use it to reproduce results

## Next Steps

After running sweeps:
1. Identify optimal parameter ranges from `summary.csv`
2. Run focused sweeps around optimal values
3. Use best configuration in `pli.ipynb` for production runs
4. Document your findings!
