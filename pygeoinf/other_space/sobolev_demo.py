"""
Demonstration of improved SobolevFunction with callable support and
mathematical point evaluation restrictions.
"""

import numpy as np
from interval_domain import IntervalDomain
from sobolev_functions import create_sobolev_function

# Create domain
domain = IntervalDomain(0, np.pi, name="[0, π]")

print("=== SobolevFunction Demonstration ===")
print(f"Domain: {domain}")
print()

# Example 1: Function defined via callable (SOLA_DLI style)
print("1. Function via callable (like SOLA_DLI):")
try:
    # This should work: s = 1.5 > 1/2
    f1 = create_sobolev_function(
        domain,
        sobolev_order=1.5,
        evaluate_callable=lambda x: np.sin(x) * np.exp(-0.1*x),
        name="sin(x)exp(-0.1x)"
    )

    # Test evaluation
    x_test = np.pi/4
    value = f1(x_test)
    print(f"  f1({x_test:.3f}) = {value:.6f}")
    print(f"  Sobolev order: {f1.sobolev_order} > 0.5 ✓")

except Exception as e:
    print(f"  Error: {e}")

print()

# Example 2: Low regularity function - should fail point evaluation
print("2. Low regularity function (s ≤ 1/2):")
try:
    # This should fail: s = 0.3 ≤ 1/2
    f2 = create_sobolev_function(
        domain,
        sobolev_order=0.3,
        evaluate_callable=lambda x: x**(0.6),  # Still continuous
        name="x^0.6"
    )

    # This should raise an error
    value = f2(np.pi/4)
    print(f"  ERROR: This should not work!")

except ValueError as e:
    print(f"  ✓ Correctly caught: {e}")
except Exception as e:
    print(f"  Unexpected error: {e}")

print()

# Example 3: Integration works even for low regularity
print("3. Integration (works for all s > 0):")
try:
    f3 = create_sobolev_function(
        domain,
        sobolev_order=0.3,  # Low regularity
        evaluate_callable=lambda x: np.abs(x - np.pi/2)**0.6,
        name="|x - π/2|^0.6"
    )

    # Integration should work even though point evaluation doesn't
    integral = f3.integrate(method='adaptive')
    print(f"  ∫ |x - π/2|^0.6 dx over [0,π] = {integral:.6f}")
    print(f"  Note: Integration works even with s = {f3.sobolev_order} ≤ 0.5")

except Exception as e:
    print(f"  Error: {e}")

print()

# Example 4: Coefficients-based function
print("4. Function via basis coefficients:")
try:
    # Create a function using Fourier coefficients
    coeffs = np.array([1.0, 0.5, 0.2, 0.1])  # Decaying coefficients
    f4 = create_sobolev_function(
        domain,
        sobolev_order=2.0,
        coefficients=coeffs,
        basis_type='fourier',
        name="Fourier series"
    )

    x_test = np.pi/3
    value = f4(x_test)
    print(f"  f4({x_test:.3f}) = {value:.6f}")
    print(f"  Based on Fourier coefficients: {coeffs}")

except Exception as e:
    print(f"  Error: {e}")

print()

# Example 5: Comparison with mathematical theory
print("5. Mathematical point evaluation threshold:")
print("   For intervals (d=1): s > d/2 = 1/2")
print("   Testing different Sobolev orders:")

test_orders = [0.3, 0.5, 0.6, 1.0, 1.5]
for s in test_orders:
    try:
        f_test = create_sobolev_function(
            domain,
            sobolev_order=s,
            evaluate_callable=lambda x: np.sin(x),
            name=f"H^{s}"
        )
        value = f_test(np.pi/4)  # Try point evaluation
        status = "✓ Works"
    except ValueError:
        status = "✗ Fails (expected)"
    except Exception as e:
        status = f"✗ Error: {e}"

    print(f"   s = {s:3.1f}: {status}")

print("\n=== Demonstration Complete ===")
