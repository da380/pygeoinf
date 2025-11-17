"""Test the main changes needed for the discontinuity notebook."""

print("=" * 70)
print("Testing notebook changes for discontinuous spaces")
print("=" * 70)

# Test 1: Import all necessary modules
print("\n1. Testing imports...")
try:
    from pygeoinf.interval.interval_domain import IntervalDomain
    from pygeoinf.interval import Lebesgue
    from pygeoinf.hilbert_space import EuclideanSpace
    from pygeoinf.interval.operators import SOLAOperator
    from pygeoinf.interval.function_providers import NormalModesProvider, BumpFunctionProvider
    from pygeoinf.interval.functions import Function
    from pygeoinf.interval.boundary_conditions import BoundaryConditions
    from pygeoinf.direct_sum import BlockDiagonalLinearOperator
    import numpy as np
    print("   ✓ All imports successful")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    exit(1)

# Test 2: Create discontinuous space
print("\n2. Creating discontinuous model space...")
function_domain = IntervalDomain(0, 1, boundary_type='open')
N = 100
R_discontinuity = 0.5
M = Lebesgue.with_discontinuities(N, function_domain, [R_discontinuity], basis=None)
print(f"   ✓ Created: {type(M).__name__} with {M.number_of_subspaces} subspaces")

# Test 3: Create full domain space for kernels
print("\n3. Creating full domain space for kernels...")
M_full = Lebesgue(N, function_domain, basis=None)
print(f"   ✓ Created: {type(M_full).__name__}")

# Test 4: Create operators
print("\n4. Creating forward and property operators...")
N_d, N_p = 20, 10
D = EuclideanSpace(N_d)
P = EuclideanSpace(N_p)

normal_modes_provider = NormalModesProvider(
    M_full,
    n_modes_range=(1, 10),
    coeff_range=(-3, 3),
    gaussian_width_percent_range=(1, 5),
    freq_range=(0.1, 10),
    random_state=42
)

width = 0.2
centers = np.linspace(function_domain.a + width / 2, function_domain.b - width / 2, N_p)
target_provider = BumpFunctionProvider(M_full, centers=centers, default_width=width)

G = SOLAOperator.for_direct_sum(M, D, normal_modes_provider)
T = SOLAOperator.for_direct_sum(M, P, target_provider)

print(f"   ✓ G: {type(G).__name__}")
print(f"   ✓ T: {type(T).__name__}")

# Test 5: Create discontinuous function
print("\n5. Creating discontinuous test function...")
# Use numpy arrays to safely handle vectorized evaluation
f_lower = Function(M.subspace(0), evaluate_callable=lambda x: np.ones_like(x, dtype=float))
f_upper = Function(M.subspace(1), evaluate_callable=lambda x: 2.0 * np.asarray(x))
m_test = [f_lower, f_upper]
print("   ✓ Created discontinuous function as list")

# Test 6: Apply operators
print("\n6. Applying operators...")
try:
    data = G(m_test)
    props = T(m_test)
    print(f"   ✓ G(m) shape: {data.shape}")
    print(f"   ✓ T(m) shape: {props.shape}")
except Exception as e:
    print(f"   ✗ Operator application failed: {e}")
    exit(1)

# Test 7: Helper function for visualization
print("\n7. Testing visualization helper...")
def eval_discontinuous(f_list, x_array):
    result = np.zeros_like(x_array)
    for i, f in enumerate(f_list):
        subdomain = f.space.function_domain
        mask = subdomain.contains(x_array)
        if np.any(mask):
            result[mask] = f.evaluate(x_array[mask])
    return result

x = np.linspace(0.01, 0.99, 100)  # Avoid boundaries
vals = eval_discontinuous(m_test, x)
print(f"   ✓ Evaluated discontinuous function over {len(x)} points")
print(f"   ✓ Value range: [{vals.min():.3f}, {vals.max():.3f}]")

print("\n" + "=" * 70)
print("SUCCESS: All notebook changes are working correctly!")
print("=" * 70)
