# Implementation Plan: Real Sensitivity Kernel Provider

## Objective

Create a `SensitivityKernelProvider` system that:
1. Loads real sensitivity kernel data from disk
2. Converts discrete kernels to continuous `Function` instances via interpolation
3. Replaces/extends `NormalModeProvider` to use real computed kernels
4. Integrates seamlessly with existing pygeoinf.interval infrastructure

---

## Phase 1: Data Loading and Parsing (Foundation)

### Task 1.1: Create Kernel Data Loader
**File:** `pygeoinf/interval/sensitivity_kernel_loader.py`

**Functionality:**
```python
class SensitivityKernelData:
    """Container for loaded kernel data for a single mode."""
    def __init__(self, mode_id: str, data_dir: Path):
        self.mode_id = mode_id  # e.g., "00s03"
        self.n, self.l = parse_mode_id(mode_id)  # Extract overtone, angular order

        # Load volumetric kernels
        self.vp_depths, self.vp_values = load_kernel_file(...)
        self.vs_depths, self.vs_values = load_kernel_file(...)
        self.rho_depths, self.rho_values = load_kernel_file(...)

        # Load discontinuity kernel
        self.topo_depths, self.topo_values = load_kernel_file(...)

        # Load metadata from header
        self.period, self.vp_ref, self.vs_ref, self.group_velocity = parse_header(...)

    @classmethod
    def from_combined_file(cls, mode_id: str, data_dir: Path):
        """Load from sens_kernels_*.dat and topo_kernels_*.dat"""
        ...

    @classmethod
    def from_individual_files(cls, mode_id: str, data_dir: Path):
        """Load from vp-sens_*.dat, vs-sens_*.dat, etc."""
        ...
```

**Key functions:**
- `load_kernel_file(filepath)`: Parse 2-column data files
- `parse_header(filepath)`: Extract metadata from combined files
- `parse_mode_id(mode_id)`: Convert "00s03" → (n=0, l=3)
- `format_mode_id(n, l)`: Convert (n=0, l=3) → "00s03"

**Error handling:**
- Missing files (graceful failure)
- Malformed data (report line numbers)
- Inconsistent depths between vp/vs/rho

### Task 1.2: Create Mode Catalog
**File:** `pygeoinf/interval/sensitivity_kernel_catalog.py`

**Functionality:**
```python
class SensitivityKernelCatalog:
    """Manage collection of all available modes."""
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.available_modes = self._scan_available_modes()
        self.prem_depths = self._load_prem_depths()

    def _scan_available_modes(self) -> List[str]:
        """Read data_list_SP12RTS or scan directory."""
        ...

    def get_mode(self, mode_id: str) -> SensitivityKernelData:
        """Load data for specific mode."""
        ...

    def get_mode_by_nl(self, n: int, l: int) -> SensitivityKernelData:
        """Load data using (n, l) notation."""
        ...

    def list_modes(self, n_min=None, n_max=None, l_min=None, l_max=None):
        """Filter available modes."""
        ...

    def get_frequency_range(self) -> Tuple[float, float]:
        """Get min/max frequencies from all modes."""
        ...
```

**Features:**
- Lazy loading (don't load all kernels at once)
- Caching (store loaded kernels in memory)
- Query interface (find modes by frequency, overtone, etc.)

### Task 1.3: Unit Tests
**File:** `tests/test_sensitivity_kernel_loader.py`

**Tests:**
- Load individual kernel files
- Load combined kernel files
- Parse headers correctly
- Handle missing files
- Mode ID parsing/formatting
- Catalog scanning and filtering

---

## Phase 2: Interpolation and Function Creation

### Task 2.1: Depth Coordinate Normalization
**File:** `pygeoinf/interval/depth_coordinates.py`

**Functionality:**
```python
class DepthCoordinateSystem:
    """Handle depth coordinate transformations."""
    EARTH_RADIUS_KM = 6371.0

    @staticmethod
    def depth_to_normalized(depth_km: np.ndarray) -> np.ndarray:
        """Convert depth [km] to normalized [0, 1]."""
        return depth_km / DepthCoordinateSystem.EARTH_RADIUS_KM

    @staticmethod
    def normalized_to_depth(normalized: np.ndarray) -> np.ndarray:
        """Convert normalized [0, 1] to depth [km]."""
        return normalized * DepthCoordinateSystem.EARTH_RADIUS_KM

    @staticmethod
    def radius_to_depth(radius_km: np.ndarray) -> np.ndarray:
        """Convert radius from center to depth from surface."""
        return DepthCoordinateSystem.EARTH_RADIUS_KM - radius_km

    @staticmethod
    def depth_to_radius(depth_km: np.ndarray) -> np.ndarray:
        """Convert depth from surface to radius from center."""
        return DepthCoordinateSystem.EARTH_RADIUS_KM - depth_km
```

**Note:** pygeoinf.interval uses [0, 1] intervals. Need consistent mapping:
- 0.0 → Surface (0 km depth)
- 1.0 → Center (6371 km depth)

### Task 2.2: Kernel Interpolator
**File:** `pygeoinf/interval/kernel_interpolator.py`

**Functionality:**
```python
from scipy.interpolate import interp1d, UnivariateSpline

class KernelInterpolator:
    """Create interpolating functions from discrete kernel data."""

    def __init__(self, depths_km: np.ndarray, values: np.ndarray,
                 method: str = 'cubic'):
        """
        Args:
            depths_km: Depth points from surface [km]
            values: Kernel values at those depths
            method: 'linear', 'cubic', 'spline'
        """
        self.depths_km = depths_km
        self.values = values
        self.method = method

        # Normalize depths to [0, 1]
        self.depths_normalized = DepthCoordinateSystem.depth_to_normalized(depths_km)

        # Create interpolator
        self._interpolator = self._create_interpolator()

    def _create_interpolator(self):
        """Create scipy interpolator."""
        if self.method == 'linear':
            return interp1d(self.depths_normalized, self.values,
                          kind='linear', fill_value=0.0, bounds_error=False)
        elif self.method == 'cubic':
            return interp1d(self.depths_normalized, self.values,
                          kind='cubic', fill_value=0.0, bounds_error=False)
        elif self.method == 'spline':
            return UnivariateSpline(self.depths_normalized, self.values,
                                   s=0, k=3, ext=1)  # ext=1 → zero outside
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate kernel at normalized depths x ∈ [0, 1]."""
        return self._interpolator(x)

    def to_function(self, lebesgue_space: LebesgueSpace) -> np.ndarray:
        """Convert to Function coefficients in given space."""
        # Use from_function with lambda wrapping the interpolator
        return lebesgue_space.from_function(lambda x: self(x))
```

**Interpolation choices:**
- **Linear:** Fast, continuous, but not smooth
- **Cubic:** Smooth (C²), good for most kernels
- **Spline:** Smoothest, but may oscillate with sparse data

**Recommendations:**
- Use cubic for volumetric kernels (vp, vs, rho)
- Handle zeros at boundaries carefully
- Consider monotonicity constraints if needed

### Task 2.3: Discontinuity Kernel Handler
**File:** `pygeoinf/interval/discontinuity_kernels.py`

**Functionality:**
```python
class DiscontinuityKernel:
    """Handle topography sensitivity at discontinuities."""

    def __init__(self, depths_km: np.ndarray, values: np.ndarray):
        """
        Args:
            depths_km: Discontinuity depths [km]
            values: Sensitivity values at each discontinuity
        """
        self.depths_km = depths_km
        self.values = values
        self.depths_normalized = DepthCoordinateSystem.depth_to_normalized(depths_km)
        self.n_discontinuities = len(depths_km)

    def to_euclidean_vector(self) -> np.ndarray:
        """Convert to vector in EuclideanSpace."""
        return self.values

    def get_discontinuity_map(self) -> Dict[float, float]:
        """Map normalized depths to sensitivity values."""
        return dict(zip(self.depths_normalized, self.values))
```

**Integration strategy:**
Two approaches:

**Option A: Direct sum (like demo_1)**
```python
M_volumetric = HilbertSpaceDirectSum([M_vp, M_vs, M_rho])
M_discontinuities = EuclideanSpace(n_discontinuities)
M_model = HilbertSpaceDirectSum([M_volumetric, M_discontinuities])
```

**Option B: Single function space with special operators**
- Represent discontinuities as delta functions
- Use point evaluation operators at discontinuity depths
- Simpler model space, but more complex operators

**Recommendation:** Use Option A for clarity and consistency with demo_1.

### Task 2.4: Unit Tests
**File:** `tests/test_kernel_interpolation.py`

**Tests:**
- Depth coordinate conversions
- Interpolation methods (linear, cubic, spline)
- Boundary behavior (extrapolation)
- Function creation in LebesgueSpace
- Discontinuity vector creation

---

## Phase 3: Sensitivity Kernel Provider

### Task 3.1: Core Provider Class
**File:** `pygeoinf/interval/sensitivity_kernel_provider.py`

**Functionality:**
```python
class SensitivityKernelProvider:
    """Provide real sensitivity kernels as Functions."""

    def __init__(self,
                 catalog: SensitivityKernelCatalog,
                 lebesgue_space: LebesgueSpace,
                 interpolation_method: str = 'cubic'):
        """
        Args:
            catalog: Catalog of available kernels
            lebesgue_space: Function space for volumetric kernels
            interpolation_method: Interpolation method for kernels
        """
        self.catalog = catalog
        self.space = lebesgue_space
        self.method = interpolation_method
        self._cache = {}  # Cache interpolated Functions

    def get_vp_kernel(self, mode_id: str) -> np.ndarray:
        """Get vp sensitivity kernel as Function coefficients."""
        cache_key = f"vp_{mode_id}"
        if cache_key not in self._cache:
            data = self.catalog.get_mode(mode_id)
            interpolator = KernelInterpolator(data.vp_depths, data.vp_values,
                                             self.method)
            self._cache[cache_key] = interpolator.to_function(self.space)
        return self._cache[cache_key]

    def get_vs_kernel(self, mode_id: str) -> np.ndarray:
        """Get vs sensitivity kernel as Function coefficients."""
        ...

    def get_rho_kernel(self, mode_id: str) -> np.ndarray:
        """Get rho sensitivity kernel as Function coefficients."""
        ...

    def get_topo_kernel(self, mode_id: str) -> DiscontinuityKernel:
        """Get topography sensitivity kernel."""
        ...

    def get_all_kernels(self, mode_id: str) -> Dict[str, np.ndarray]:
        """Get all kernels for a mode."""
        return {
            'vp': self.get_vp_kernel(mode_id),
            'vs': self.get_vs_kernel(mode_id),
            'rho': self.get_rho_kernel(mode_id),
            'topo': self.get_topo_kernel(mode_id)
        }

    def create_forward_operator(self,
                                 mode_ids: List[str],
                                 include_discontinuities: bool = True
                                ) -> LinearOperator:
        """Create forward operator G mapping model to mode frequencies."""
        ...
```

### Task 3.2: Operator Construction
**Functionality:**
```python
def create_forward_operator(self, mode_ids, include_discontinuities=True):
    """
    Create G: [vp, vs, rho, topo] → [δω_1/ω_1, ..., δω_N/ω_N]

    For each mode i:
        δω_i/ω_i = ∫ K_vp^i · δvp/vp + ∫ K_vs^i · δvs/vs +
                   ∫ K_rho^i · δρ/ρ + Σ K_topo^i · δd

    Implemented as:
        G_i = [<K_vp^i, ·>, <K_vs^i, ·>, <K_rho^i, ·>, K_topo^i]
    """
    n_modes = len(mode_ids)

    # Create row operators for each mode
    rows = []
    for mode_id in mode_ids:
        kernels = self.get_all_kernels(mode_id)

        # Inner product operators for volumetric parameters
        G_vp = self.space.inner_product_operator(kernels['vp'])
        G_vs = self.space.inner_product_operator(kernels['vs'])
        G_rho = self.space.inner_product_operator(kernels['rho'])

        if include_discontinuities:
            # Point evaluation at discontinuity depths
            topo_kernel = kernels['topo']
            G_topo = create_discontinuity_operator(topo_kernel)

            # Combine as RowLinearOperator
            row = RowLinearOperator([[G_vp, G_vs, G_rho], [G_topo]])
        else:
            row = RowLinearOperator([G_vp, G_vs, G_rho])

        rows.append(row)

    # Stack rows
    return ColumnLinearOperator(rows)
```

**Note:** This mimics the structure in demo_1 but with real kernels.

### Task 3.3: Compatibility with Existing Infrastructure

**Integration points:**

1. **Replace NormalModeProvider:**
```python
# Old way (synthetic kernels)
from pygeoinf.interval import NormalModeProvider
provider = NormalModeProvider(interval, radial_config)
G = provider.create_operator(mode_list)

# New way (real kernels)
from pygeoinf.interval import SensitivityKernelProvider, SensitivityKernelCatalog
catalog = SensitivityKernelCatalog(data_dir)
provider = SensitivityKernelProvider(catalog, lebesgue_space)
G = provider.create_forward_operator(mode_list)
```

2. **Common interface:**
Both should implement:
```python
class KernelProviderProtocol:
    def get_vp_kernel(self, mode_id) -> np.ndarray: ...
    def get_vs_kernel(self, mode_id) -> np.ndarray: ...
    def get_rho_kernel(self, mode_id) -> np.ndarray: ...
    def create_forward_operator(self, mode_ids) -> LinearOperator: ...
```

### Task 3.4: Unit Tests
**File:** `tests/test_sensitivity_kernel_provider.py`

**Tests:**
- Kernel retrieval and caching
- Function space consistency
- Operator construction
- Inner products match expected values
- Discontinuity operator correctness

---

## Phase 4: Examples and Validation

### Task 4.1: Create Example Notebook
**File:** `pygeoinf/interval/demos/paper_demos/demo_real_kernels.ipynb`

**Contents:**
1. **Load real kernel data**
   - Initialize catalog
   - Explore available modes
   - Visualize kernel shapes

2. **Compare with synthetic kernels**
   - Plot real vs synthetic for same mode
   - Quantify differences
   - Understand discrepancies

3. **Create forward problem**
   - Set up model space (vp, vs, rho, topo)
   - Create forward operator from real kernels
   - Generate synthetic data

4. **Run inference**
   - Set up prior
   - Run LinearBayesian
   - Visualize posterior

5. **Sensitivity study**
   - Compare inference with real vs synthetic kernels
   - Examine resolution differences
   - Understand impact on uncertainty quantification

### Task 4.2: Validation Tests
**File:** `tests/test_real_kernel_validation.py`

**Tests:**
1. **Orthogonality check:** Different modes should have different kernels
2. **Normalization:** Check kernel integrals match expected values
3. **Symmetry:** Certain modes should have symmetric kernels
4. **Discontinuity consistency:** Check CMB, 660, Moho sensitivities
5. **Frequency scaling:** Higher frequency modes should have shallower penetration

### Task 4.3: Performance Benchmarks
**File:** `benchmarks/benchmark_kernel_loading.py`

**Measurements:**
- Loading time per mode
- Interpolation time
- Function creation time
- Operator construction time
- Memory usage

**Optimization targets:**
- < 10 ms per kernel load
- < 1 ms per interpolation
- Support 100+ modes without memory issues

---

## Phase 5: Documentation and Integration

### Task 5.1: API Documentation
**Files to document:**
- `sensitivity_kernel_loader.py`
- `sensitivity_kernel_catalog.py`
- `sensitivity_kernel_provider.py`
- `kernel_interpolator.py`
- `discontinuity_kernels.py`
- `depth_coordinates.py`

**Documentation style:**
- Docstrings for all public methods
- Type hints throughout
- Usage examples in docstrings
- Cross-references to related functions

### Task 5.2: User Guide
**File:** `docs/sensitivity_kernels.md`

**Sections:**
1. Overview of sensitivity kernels
2. Data structure and format
3. Installation and setup
4. Basic usage examples
5. Advanced topics (custom interpolation, discontinuities)
6. Troubleshooting
7. References

### Task 5.3: Tutorial Notebook
**File:** `pygeoinf/interval/demos/tutorials/real_sensitivity_kernels_tutorial.ipynb`

**Progressive tutorial:**
1. What are sensitivity kernels?
2. Loading and visualizing real kernels
3. Creating Functions from kernel data
4. Building forward operators
5. Running inference with real kernels
6. Comparing with synthetic kernels

### Task 5.4: Integration with Existing Demos
**Modifications needed:**
- Update demo_1.ipynb to optionally use real kernels
- Create demo_1_real_kernels.ipynb as variant
- Add to demo_1_config.py: `use_real_kernels: bool = False`
- Modify demo_1_experiment.py to support both modes

---

## Phase 6: Advanced Features (Optional)

### Task 6.1: Kernel Caching System
**File:** `pygeoinf/interval/kernel_cache.py`

**Features:**
- Disk caching of interpolated Functions
- HDF5 storage for fast loading
- Cache invalidation when parameters change
- Memory-mapped arrays for large datasets

### Task 6.2: Kernel Visualization Tools
**File:** `pygeoinf/interval/kernel_visualization.py`

**Features:**
- Plot kernels vs depth
- Compare multiple modes
- Animated depth slices
- 2D kernel comparison plots
- Export publication-quality figures

### Task 6.3: Kernel Filtering and Selection
**File:** `pygeoinf/interval/kernel_selection.py`

**Features:**
- Select modes by frequency band
- Filter by depth sensitivity
- Find modes sensitive to target depth
- Optimal mode selection for inversion

### Task 6.4: Anisotropic Kernels (Future)
**Note:** Current data is isotropic (_iso.dat). If anisotropic data becomes available:
- Extend SensitivityKernelData for additional parameters
- Support kernels for Vsh, Vph, Vsv, Vpv, η
- Update operators for anisotropic parameters

---

## Implementation Timeline

### Phase 1: Foundation (1-2 weeks)
- Data loader
- Catalog system
- Unit tests

### Phase 2: Interpolation (1 week)
- Coordinate systems
- Interpolators
- Discontinuity handling

### Phase 3: Provider (1-2 weeks)
- Core provider class
- Operator construction
- Integration with existing code

### Phase 4: Validation (1 week)
- Example notebook
- Validation tests
- Performance benchmarks

### Phase 5: Documentation (1 week)
- API docs
- User guide
- Tutorials

### Phase 6: Advanced (ongoing)
- Caching
- Visualization
- Selection tools

**Total estimated time:** 5-8 weeks for core functionality

---

## Success Criteria

### Functional Requirements
✓ Load all 142 modes without errors
✓ Create interpolated Functions matching LebesgueSpace
✓ Build forward operators compatible with LinearBayesian
✓ Handle discontinuities correctly
✓ Cache for performance

### Quality Requirements
✓ >95% test coverage
✓ Comprehensive documentation
✓ < 100 ms to load and interpolate one mode
✓ Memory efficient (< 1 GB for 100 modes)

### Scientific Requirements
✓ Interpolation error < 1% of kernel magnitude
✓ Conservation of kernel integrals
✓ Consistent with seismological expectations
✓ Validated against published results (if available)

---

## Open Questions

1. **Interpolation method:** Linear, cubic, or spline? Benchmark needed.

2. **Discontinuity representation:** Direct sum or embedded?

3. **Normalization:** Are kernels already normalized? Need to verify.

4. **Missing modes:** How to handle requests for modes not in dataset?

5. **Anisotropy:** Future-proof design for anisotropic kernels?

6. **Coordinate convention:** Confirm depth vs radius conventions in all files.

7. **Integration with NormalModeProvider:** Replace or extend?

---

## Dependencies

### New Dependencies (if not already present)
- `scipy`: For interpolation
- `h5py`: For caching (optional)

### Internal Dependencies
- `pygeoinf.interval.Interval`
- `pygeoinf.interval.LebesgueSpace`
- `pygeoinf.interval.EuclideanSpace`
- `pygeoinf.interval.HilbertSpaceDirectSum`
- `pygeoinf.linear_operators.LinearOperator`
- `pygeoinf.linear_operators.RowLinearOperator`

---

## Risk Mitigation

### Risk 1: Interpolation Artifacts
**Mitigation:**
- Benchmark multiple interpolation methods
- Validate against known properties
- Add smoothing if needed

### Risk 2: Performance Issues
**Mitigation:**
- Profile early
- Implement caching
- Use lazy loading

### Risk 3: API Incompatibility
**Mitigation:**
- Design common interface from start
- Extensive integration tests
- Deprecation path for old API

### Risk 4: Data Quality Issues
**Mitigation:**
- Validation tests for physical reasonableness
- Cross-check with published kernels
- Document known issues

---

## Next Steps

1. **Immediate:** Review and approve this implementation plan
2. **Week 1:** Implement data loader (Task 1.1-1.3)
3. **Week 2:** Implement interpolation (Task 2.1-2.4)
4. **Week 3-4:** Implement provider (Task 3.1-3.4)
5. **Week 5:** Create examples and validation (Task 4.1-4.3)
6. **Week 6:** Documentation (Task 5.1-5.4)

---

## References

- PREM: Dziewonski & Anderson (1981), PEPI
- Normal mode theory: Dahlen & Tromp (1998), Theoretical Global Seismology
- Sensitivity kernels: Li & Romanowicz (1995), GJI
