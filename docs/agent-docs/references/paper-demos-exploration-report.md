# Paper Demos Exploration Report

**Date:** 2025-01-XX
**Agent:** Explorer
**Scope:** Full analysis of `intervalinf/demos/old_demos/paper_demos/` notebooks, `kernel_utils.py`, and API mapping against current `intervalinf` and `pygeoinf` packages.

---

## 1. Current intervalinf API Summary

### Top-level exports (`intervalinf/__init__.py`)

| Symbol | Type | Description |
|--------|------|-------------|
| `IntervalDomain` | Core | Bounded 1D domain `[a, b]` |
| `BoundaryConditions` | Core | BC specification (dirichlet/neumann/mixed) |
| `Function` | Core | Function on a domain or space |
| `IntegrationConfig` | Config | Base numerical integration config |
| `ParallelConfig` | Config | Parallelism settings |
| `Lebesgue` | Space | $L^2$ Hilbert space on interval; subclasses `pygeoinf.HilbertSpace` |
| `LebesgueSpaceDirectSum` | Space | Direct sum of Lebesgue spaces |
| `LebesgueIntegrationConfig` | Config | Integration settings for Lebesgue spaces |
| `LebesgueParallelConfig` | Config | Parallel settings for Lebesgue spaces |
| `Sobolev` | Space | $H^s$ space; subclasses `pygeoinf.MassWeightedHilbertSpace` |
| `SobolevSpaceDirectSum` | Space | Direct sum of Sobolev spaces |
| `LinearFormKernel` | Form | Linear form from kernel function; subclasses `pygeoinf.LinearForm` |
| `KnownRegion` | Special | Marks a sub-interval as having a known function value |
| `PartitionedLebesgueSpace` | Special | Lebesgue space with known/unknown partitions |

### Subpackage: `intervalinf.operators`

| Symbol | Module | Description |
|--------|--------|-------------|
| `SOLAOperator` | `sola.py` | Subtractive Optimally Localized Averaging operator |
| `Laplacian` | `laplacian.py` | Laplacian operator (spectral/FD methods) |
| `BesselSobolev` | `bessel.py` | $(I - \alpha\Delta)^{s/2}$ operator |
| `BesselSobolevInverse` | `bessel.py` | Inverse Bessel-Sobolev (prior covariance) |
| `Gradient` | `gradient.py` | Finite-difference gradient operator |
| `SpectralOperator` | `base.py` | ABC for spectral operators; subclasses `pygeoinf.LinearOperator` |

### Subpackage: `intervalinf.providers`

| Symbol | Module | Description |
|--------|--------|-------------|
| `NormalModesProvider` | `functions/data.py` | Gaussian-modulated trig functions (synthetic kernels) |
| `BumpFunctionProvider` | `functions/smooth.py` | $C^\infty$ bump functions with compact support |
| `NullFunctionProvider` | `base.py` | Returns zero functions |
| `IndexedFunctionProvider` | `base.py` | ABC for indexed function providers |
| `SineFunctionProvider` | `functions/trigonometric.py` | Sine basis functions |
| `CosineFunctionProvider` | `functions/trigonometric.py` | Cosine basis functions |
| `FourierFunctionProvider` | `functions/trigonometric.py` | Fourier (sin+cos) basis |
| `KernelProvider` | `functions/data.py` | Loads kernel functions from data files |

### Subpackage: `intervalinf.sampling`

| Symbol | Module | Description |
|--------|--------|-------------|
| `KLSampler` | `kl_sampler.py` | Karhunen-Loève expansion sampler |

---

## 2. pygeoinf API Summary (as used by paper_demos)

All imports come from the top-level `pygeoinf` package:

| Symbol | Module | Description |
|--------|--------|-------------|
| `EuclideanSpace` | `hilbert_space.py` | Finite-dimensional Euclidean space $\mathbb{R}^n$ |
| `HilbertSpaceDirectSum` | `direct_sum.py` | Direct sum $\mathcal{H}_1 \oplus \mathcal{H}_2 \oplus \cdots$ |
| `LinearOperator` | `linear_operators.py` | ABC for linear operators between Hilbert spaces |
| `LinearForm` | `linear_forms.py` | ABC for linear functionals |
| `RowLinearOperator` | `direct_sum.py` | Operator built from row of blocks $[G_1 \; G_2 \; \cdots]$ |
| `GaussianMeasure` | `gaussian_measure.py` | Gaussian measure with covariance + expectation + sampler |
| `LinearForwardProblem` | `forward_problem.py` | Forward problem $d = G(m) + \epsilon$ |
| `LinearBayesianInversion` | `linear_bayesian.py` | Bayesian linear inversion: prior + forward → posterior |
| `CholeskySolver` | `linear_solvers.py` | Direct Cholesky solver for posterior computation |

### Key pygeoinf API patterns used in paper_demos

```python
# Direct sum of spaces
M_model = HilbertSpaceDirectSum([M_functions, M_euclidean])

# Forward operator from blocks
G = RowLinearOperator([G_vp, G_vs, G_rho, G_sigma_0, G_sigma_1], domain=M_model, range=D)

# Gaussian measure construction
M_prior_vp = GaussianMeasure(covariance=C_0_vp, expectation=m_0_vp, sample=sampler_vp.sample)
M_prior_sigma = GaussianMeasure.from_covariance_matrix(M_sigma, cov_matrix, expectation=mean)
M_prior = GaussianMeasure.from_direct_sum([M_prior_vp, M_prior_vs, ...])

# Bayesian inference
forward_problem = LinearForwardProblem(G, data_error_measure=gaussian_D_noise)
bayesian_inference = LinearBayesianInversion(forward_problem, M_prior)
solver = CholeskySolver(parallel=True, n_jobs=12)
posterior = bayesian_inference.model_posterior_measure(d_tilde, solver)

# Property posterior via affine mapping
property_posterior = posterior.affine_mapping(operator=T)
cov_P = property_posterior.covariance.matrix(dense=True, parallel=True, n_jobs=8)
```

---

## 3. Per-Notebook Analysis

### 3.1 `demo_1.ipynb` — Synthetic Normal Modes Example

**Purpose:** Full synthetic inversion using normal-mode kernels. Demonstrates the complete LinearBayesianInversion pipeline with 3 function spaces ($\delta v_p$, $\delta v_s$, $\delta\rho$) + 2 Euclidean parameters ($\delta\Sigma^0$, $\delta\Sigma^1$).

**Structure:** ~1174 lines, 20+ cells. Imports → Domain setup → Space construction → Kernel providers → Forward operator → True model → Synthetic data → Prior (BesselSobolevInverse + KLSampler) → Bayesian inference → Model posterior → Property posterior → Visualization.

**Import block:**
```python
from intervalinf import (
    Lebesgue, IntervalDomain, LebesgueIntegrationConfig, IntegrationConfig,
    ParallelConfig, LebesgueSpaceDirectSum, Function, BoundaryConditions
)
from intervalinf.operators import SOLAOperator, Laplacian, BesselSobolevInverse
from intervalinf.sampling import KLSampler
from intervalinf.providers import NormalModesProvider, BumpFunctionProvider, NullFunctionProvider
from pygeoinf import (
    EuclideanSpace, RowLinearOperator, GaussianMeasure,
    LinearForwardProblem, LinearBayesianInversion, CholeskySolver,
    HilbertSpaceDirectSum, LinearOperator, LinearForm
)
```

**Key API patterns:**
- `IntervalDomain(0, 1)` — unit interval
- `Lebesgue(N, domain, basis='sine', ...)` — finite-dim $L^2$ with sine basis
- `BoundaryConditions(bc_type='dirichlet')` — Dirichlet BCs
- `Laplacian(M, bcs, alpha, method='spectral', dofs=100, ...)` — Laplacian with spectral method
- `BesselSobolevInverse(M, M, k, s, L, dofs=512, ...)` — prior covariance operator
- `KLSampler(C_0, mean=m_0, n_modes=K)` — KL sampler
- `NormalModesProvider(space)` → `provider.get_function_by_index(i)` — synthetic kernels
- `SOLAOperator.from_provider(...)` — build SOLA operator from provider
- `Function(space, evaluate_callable=lambda x: ...)` — construct functions
- Two workflows: full model posterior (with dense covariance, sampling) vs. direct property posterior (fast)

**Potential issues:** None — all imports resolve against current API. Uses `Lebesgue` from top-level (works, it's in `__init__.py`).

---

### 3.2 `demo_1_realistic.ipynb` — Realistic Seismic Kernels

**Purpose:** Same structure as demo_1 but with real sensitivity kernels loaded from `.dat` files via `kernel_utils.py`. Domain is $[0, R_\oplus]$ (Earth radius in km).

**Structure:** ~1196 lines. Same pipeline as demo_1 but uses `SensitivityKernelCatalog` and `SensitivityKernelProvider` from `kernel_utils.py` instead of `NormalModesProvider`.

**Import block:** Identical to demo_1, plus:
```python
from kernel_utils import SensitivityKernelCatalog, SensitivityKernelProvider, EARTH_RADIUS_KM
```

**Key differences from demo_1:**
- Domain: `IntervalDomain(0, EARTH_RADIUS_KM)` instead of `[0, 1]`
- Basis: `'cosine'` instead of `'sine'`
- Integration: uses `'trapz'` method
- Kernels: loaded from `'../kernels_modeplotaat_Adrian'` directory via `SensitivityKernelCatalog`
- All 3 function spaces (vp, vs, rho) are standard Lebesgue (no partitioning)

**Potential issues:** None — uses current API. `SensitivityKernelProvider` in kernel_utils.py already updated to use new-style `Function(space, evaluate_callable=...)` and `IndexedFunctionProvider` ABC.

---

### 3.3 `demo_1_realistic_vs_known.ipynb` — Realistic with Known Regions

**Purpose:** Extends demo_1_realistic by using `PartitionedLebesgueSpace` for $\delta v_s$, incorporating the constraint that shear velocity is zero in the outer core.

**Structure:** ~1375 lines. Same pipeline but with partitioned model space for vs.

**Import block:** Same as demo_1, plus:
```python
from intervalinf import KnownRegion, PartitionedLebesgueSpace
from kernel_utils import SensitivityKernelCatalog, SensitivityKernelProvider, EARTH_RADIUS_KM
```

**Key API patterns unique to this notebook:**
- `KnownRegion.zero(domain_OC)` — outer core has known zero shear velocity
- `PartitionedLebesgueSpace(...)` — Lebesgue space with known + unknown partitions
- Separate `Laplacian` instances for inner core and mantle sub-domains
- Separate `BesselSobolevInverse` for each unknown partition
- `GaussianMeasure.from_direct_sum([M_prior_vs_IC, M_prior_vs_M])` — combines sub-priors
- Basis: `'ND'` for inner core (Neumann-Dirichlet), `'cosine'` for mantle
- Unpacking: `(sample_vs_IC, sample_vs_M) = sample_vs` — direct-sum decomposition

**Boundary constants used:**
```python
ICB_RADIUS = 1221.5   # Inner Core Boundary
CMB_RADIUS = 3480.0   # Core-Mantle Boundary
```

**Potential issues:** None — all imports resolve. This is the template for demo_2 and wrong_prior.

---

### 3.4 `demo_2.ipynb` — CMB Jump Inference (N_p=1)

**Purpose:** Infers a single scalar property (CMB discontinuity jump) using the same multi-space model as demo_1_realistic_vs_known. The property space is 1-dimensional ($N_p = 1$).

**Structure:** ~1202 lines. Nearly identical to demo_1_realistic_vs_known.

**Key difference:** The target operator $T$ maps to a 1D Euclidean property space (single CMB jump value), rather than a multi-point SOLA target.

**Potential issues:** None.

---

### 3.5 `wrong_prior.ipynb` — Misspecified Prior

**Purpose:** Demonstrates inference with intentionally wrong prior hyperparameters (overly smooth prior). Same model structure as demo_1_realistic_vs_known.

**Structure:** ~1375 lines. Identical to demo_1_realistic_vs_known except for hyperparameters.

**Key differences (hyperparameters only):**
```python
# "Wrong" prior (overly smooth)
s_vp = 6.0              # vs. 4.0 in correct prior
overall_variance_vp = np.power(10, 1.5)  # vs. np.power(10, 1.0)
s_vs = 6.0              # vs. 4.0 in correct prior
overall_variance_vs = np.power(10, 3)    # vs. np.power(10, 1.0)
```

**Potential issues:** None.

---

## 4. kernel_utils.py Analysis

**Location:** `intervalinf/demos/old_demos/paper_demos/kernel_utils.py`
**Status:** **Already updated to current API**

### Contents

| Class/Constant | Description |
|----------------|-------------|
| `EARTH_RADIUS_KM` | Earth radius constant (6371.0 km) |
| `DepthCoordinateSystem` | Converts between depth and radius coordinates |
| `TopoKernel` | Simple topographic kernel (discontinuity sensitivity) |
| `SensitivityKernelCatalog` | Loads and organizes `.dat` kernel files from disk |
| `SensitivityKernelProvider` | `IndexedFunctionProvider` subclass — serves kernels as `Function` objects |

### API usage verification

```python
# Already uses new-style Function construction
Function(space, evaluate_callable=_eval)

# Already subclasses IndexedFunctionProvider correctly
class SensitivityKernelProvider(IndexedFunctionProvider):
    def __init__(self, space, catalog, ...):
        super().__init__(space)  # ← correct new-style ABC init

# Uses self._space_or_domain (internal attribute) — works with current API
```

**No changes needed.** kernel_utils.py is compatible with the current intervalinf API.

---

## 5. Subdirectory Status

All 6 subdirectories contain **only output files** (PNG/PDF images). No notebooks or Python files.

| Directory | Contents |
|-----------|----------|
| `example_1/` | Output plots from demo_1 |
| `example_1_vs_known/` | Output plots from demo_1_realistic_vs_known |
| `example_2/` | Output plots from demo_2 |
| `example_3_high_rez/` | High-resolution output plots |
| `example_3_high_rez_long/` | Extended high-resolution output |
| `example_3_low_rez/` | Low-resolution output plots |

---

## 6. Old → New API Mapping

### Import path differences

The paper_demos import `Lebesgue` from the top-level package, while the updated demos import from the subpackage:

| Paper Demos (current) | Updated Demos (preferred) | Status |
|----------------------|--------------------------|--------|
| `from intervalinf import Lebesgue` | `from intervalinf.spaces import Lebesgue` | **Both work** — `Lebesgue` is in `__init__.py` |
| `from intervalinf import LebesgueSpaceDirectSum` | `from intervalinf.spaces import LebesgueSpaceDirectSum` | **Both work** |
| `from intervalinf import Function` | `from intervalinf import Function` | **Same** |
| `from intervalinf import IntervalDomain` | `from intervalinf import IntervalDomain` | **Same** |
| `from intervalinf import BoundaryConditions` | `from intervalinf import BoundaryConditions` | **Same** |
| `from intervalinf import KnownRegion` | `from intervalinf import KnownRegion` | **Same** |
| `from intervalinf import PartitionedLebesgueSpace` | `from intervalinf import PartitionedLebesgueSpace` | **Same** |
| `from intervalinf.operators import SOLAOperator, Laplacian, BesselSobolevInverse` | Same | **Same** |
| `from intervalinf.sampling import KLSampler` | Same | **Same** |
| `from intervalinf.providers import NormalModesProvider, ...` | Same | **Same** |

### API pattern differences

| Pattern | Paper Demos | Updated Demos | Breaking? |
|---------|------------|---------------|-----------|
| Function creation | `Function(space, evaluate_callable=...)` | `Function(domain, evaluate_callable=...)` | **No** — both accepted |
| Lebesgue construction | `Lebesgue(N, domain, basis='sine', ...)` | `Lebesgue(N, domain, basis='fourier')` | **No** — same API |
| Space attachment | Implicit (function created with space) | `f.is_attached` property | **No** — additive |

### Key finding

**The paper_demos are already using the current intervalinf API.** There are no broken imports or deprecated patterns. The notebooks will run as-is against the current codebase.

The only optional improvement would be to update import paths from `from intervalinf import Lebesgue` to `from intervalinf.spaces import Lebesgue` for consistency with the newer demo style, but this is cosmetic — both paths resolve correctly.

---

## 7. Summary & Recommendations

1. **All 5 paper_demos use a consistent, valid API** — no updates required for correctness
2. **kernel_utils.py is already migrated** to the current `Function`/`IndexedFunctionProvider` API
3. **No notebooks exist in subdirectories** — they are output-only folders
4. **The shared import block** across all 5 notebooks is nearly identical; consider extracting to a shared setup module if refactoring
5. **Optional cosmetic changes:**
   - Standardize `from intervalinf.spaces import Lebesgue` (preferred import path)
   - Move `kernel_utils.py` into `intervalinf.providers.functions.data` if it becomes reusable beyond demos
