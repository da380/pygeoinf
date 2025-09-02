"""Debug the Lebesgue space issues."""

import numpy as np
from pygeoinf.interval.lebesgue_space import Lebesgue
from pygeoinf.interval.interval_domain import IntervalDomain

# Create a simple Lebesgue space
domain = IntervalDomain(0, 1)
callables = [
    lambda x: np.ones_like(x),  # constant
    lambda x: x,                # linear
    lambda x: x**2              # quadratic
]
space = Lebesgue(3, domain, basis=callables)

print("=== Debug Lebesgue Space Issues ===")

# Test basic function creation
print("\n1. Testing random function generation:")
x = space.random()
print(f"   Random function: {type(x)}")
print(f"   Has coefficients: {hasattr(x, 'coefficients')}")
if hasattr(x, 'coefficients'):
    print(f"   Coefficients: {x.coefficients}")

# Test copying
print("\n2. Testing copy operation:")
x_copy = space.copy(x)
print(f"   Copy: {type(x_copy)}")
print(f"   Copy has coefficients: {hasattr(x_copy, 'coefficients')}")
if hasattr(x_copy, 'coefficients'):
    print(f"   Copy coefficients: {x_copy.coefficients}")

# Test multiplication
print("\n3. Testing multiplication:")
a = 2.0
expected = space.multiply(a, x)
print(f"   Expected result: {type(expected)}")
if hasattr(expected, 'coefficients'):
    print(f"   Expected coefficients: {expected.coefficients}")

# Test in-place operation
print("\n4. Testing in-place ax operation:")
print(f"   Before ax - x_copy coefficients: {x_copy.coefficients}")
space.ax(a, x_copy)
print(f"   After ax - x_copy coefficients: {x_copy.coefficients}")

# Test to_components
print("\n5. Testing to_components:")
expected_components = space.to_components(expected)
copy_components = space.to_components(x_copy)
print(f"   Expected components: {expected_components}")
print(f"   Copy components: {copy_components}")
print(f"   Are equal: {np.allclose(expected_components, copy_components)}")
