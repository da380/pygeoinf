# Sensitivity Kernel Data Structure Analysis

## Overview

This directory contains **real sensitivity kernel data** for seismic normal modes computed on the PREM Earth model. The data represents discrete sensitivity kernels for:
- **vp**: P-wave velocity
- **vs**: S-wave velocity
- **rho**: Density
- **topo**: Topography (discontinuity depth)

## Directory Structure

### File Organization

The data is organized by **normal mode** identified as `{n}s{l}` where:
- `n`: Overtone number (00-19)
- `l`: Angular order (01-30)

For each mode, there are multiple file types:

#### 1. Individual Sensitivity Files (4 per mode)
- `vp-sens_{n}s{l}_iso.dat`: P-wave velocity sensitivity (~222 lines)
- `vs-sens_{n}s{l}_iso.dat`: S-wave velocity sensitivity (~222 lines)
- `rho-sens_{n}s{l}_iso.dat`: Density sensitivity (~222 lines)
- `topo-sens_{n}s{l}_iso.dat`: Topography sensitivity (~7 lines, discontinuities only)

#### 2. Combined Sensitivity Files (2 per mode)
- `sens_kernels_{n}s{l}_iso.dat`: Combined vp, vs, rho kernels (~224 lines)
- `topo_kernels_{n}s{l}_iso.dat`: Topography kernel with header (~9 lines)

### File Formats

#### Individual Sensitivity Files (vp-sens, vs-sens, rho-sens)
**Format:** 2 columns, space-separated
```
depth(km)  sensitivity_value
```

**Example (vp-sens_00s03_iso.dat):**
```
0.000 0.00000E+00
22.620 0.00000E+00
45.240 0.00000E+00
...
```

**Characteristics:**
- ~222 depth points from surface (0 km) to core
- Values in scientific notation (Fortran-style `E` notation)
- Depth is measured from **Earth's surface downward**

#### Topography Sensitivity Files (topo-sens)
**Format:** 2 columns, space-separated
```
depth(km)  sensitivity_value
```

**Example (topo-sens_00s03_iso.dat):**
```
5149.509 -0.10027E-01
2891.042 -0.39741E+01
670.000 0.86138E-01
400.000 0.33767E-01
220.000 0.15154E+00
24.371 0.29583E+00
3.000 -0.67680E+00
```

**Characteristics:**
- Only ~7 values corresponding to **major discontinuities** in PREM
- Depths: ~5150 km (D"), ~2891 km (CMB), ~670 km (660 discontinuity), ~400 km, ~220 km, ~24 km (Moho), ~3 km
- Sparse representation (discontinuities only)

#### Combined Sensitivity Files (sens_kernels)
**Format:** 4 columns with 2-line header
```
#period(s)   vp_ref   vs_ref
#   ??       group_velocity
depth(km)  vp_sens  vs_sens  rho_sens
```

**Example (sens_kernels_00s03_iso.dat):**
```
#2134.18042   6.70531   5.35906
#   0.46856 417.48904
   0.000  0.00000E+00  0.00000E+00  0.00000E+00
  22.620  0.00000E+00  0.00000E+00  0.63000E-04
  45.240  0.00000E+00  0.00000E+00  0.25199E-03
...
```

**Header information:**
- Line 1: `#period(s)  vp_ref  vs_ref` - Mode period and reference velocities
- Line 2: `#  ??  group_velocity` - Additional mode properties
- Data: depth, vp sensitivity, vs sensitivity, rho sensitivity

#### Topography Combined Files (topo_kernels)
**Format:** Same as topo-sens but with header
```
#period(s)   vp_ref   vs_ref
#   ??       group_velocity
depth(km)  topo_sens
```

### Reference Files

#### `data_list_SP12RTS`
**Purpose:** Master list of all available modes
**Format:** One mode identifier per line
```
00s03
00s04
...
19s11
```

**Total modes:** 142 modes

#### `PREM_depth_layers_all`
**Purpose:** PREM model depth discretization
**Format:** 2 columns
```
layer_index  depth(km)
```

**Characteristics:**
- 222 layers from surface to center
- Layer 1: 6371.0 km (surface, Earth radius)
- Layer 222: 0.0 km (center)
- **Note:** Depths are given as radii (distance from Earth's center)
- To convert to depth from surface: `depth = 6371.0 - radius`

## Data Characteristics

### Depth Coordinate Systems

**Two conventions are mixed:**

1. **In sensitivity files:** Depth from surface (0 = surface, increases downward)
2. **In PREM_depth_layers_all:** Radius from center (6371 = surface, decreases inward)

**Conversion:**
```python
depth_from_surface = 6371.0 - radius_from_center
```

### Discretization

- **Volumetric kernels (vp, vs, rho):** ~222 depth points (fine discretization)
- **Discontinuity kernels (topo):** ~7 depth points (major discontinuities only)
- **Depth spacing:** Non-uniform, denser near surface and discontinuities

### Normal Mode Coverage

**Overtone numbers:** 0-19 (20 overtone families)
**Angular orders:** Varies by overtone (1-30)
**Total modes:** 142

**Mode naming convention:**
- `00s03`: Fundamental mode (n=0), l=3
- `01s05`: 1st overtone (n=1), l=5
- `19s11`: 19th overtone (n=19), l=11

### Physical Units

**Sensitivity kernels represent:**
- δω/ω per unit relative perturbation in model parameter
- Dimensionless if perturbations are relative (δm/m)
- Units depend on normalization convention used

**Typical interpretation:**
```
δω/ω = ∫ K_vp(r) δvp/vp dr + ∫ K_vs(r) δvs/vs dr + ∫ K_rho(r) δρ/ρ dr + Σ K_topo(d_i) δd_i
```

## Data Quality Notes

### Completeness
- Not all (n,l) combinations present (only 142 out of potential hundreds)
- Selection based on SP12RTS tomography model usage
- Higher overtones (n>19) not included

### Precision
- Values stored in Fortran scientific notation (e.g., `0.12345E-03`)
- Typical precision: 5 significant figures

### Zeros
- Many near-surface vp/vs values are exactly zero (modes don't sense shallow structure)
- This is physically correct for long-period modes

---

## Proposed Usage in pygeoinf

### Current State
The `NormalModeProvider` currently generates **synthetic** sensitivity kernels using analytical formulas. These are approximations.

### Proposed Enhancement
Replace synthetic kernels with **real computed kernels** from this dataset via interpolation.

### Key Advantages
1. **Accuracy:** Real kernels capture Earth's actual structure (PREM)
2. **Discontinuities:** Properly accounts for major boundaries (Moho, 660, CMB, etc.)
3. **Validation:** Can validate inference algorithms against real seismological data

### Implementation Considerations

#### Interpolation Strategy
- **Volumetric kernels (vp, vs, rho):** Use spline interpolation on 222 points
- **Topography kernels:** Handle as discrete jumps at discontinuity depths
- **Depth coordinate:** Convert PREM radii to depths from surface for consistency

#### Function Creation
Transform discrete kernel data into `Function` instances:
```python
# Pseudocode
depths = kernel_data[:, 0]
values = kernel_data[:, 1]
# Convert to normalized depth [0, 1]
normalized_depths = depths / 6371.0
# Create interpolating function
kernel_function = create_interpolated_function(normalized_depths, values)
```

#### Mode Selection
- Use `data_list_SP12RTS` to identify available modes
- Map (n, l) to file names: `{n:02d}s{l:02d}`
- Handle missing modes gracefully

#### Discontinuity Handling
Two options:
1. **Separate spaces:** Use direct sum with function space + discontinuity space (like demo_1)
2. **Embedded:** Treat discontinuities as part of volumetric inversion with special operators

---

## Action Items (Next Steps)

See accompanying `IMPLEMENTATION_PLAN.md` for detailed action plan.
