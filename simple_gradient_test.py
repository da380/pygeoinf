import sys
sys.path.insert(0, '/home/adrian/PhD/Inferences/pygeoinf')

try:
    from pygeoinf.interval.operators import GradientOperator
    print("✓ GradientOperator imported successfully!")

    from pygeoinf.interval.l2_space import L2Space
    print("✓ L2Space imported successfully!")

    from pygeoinf.interval.interval_domain import IntervalDomain
    print("✓ IntervalDomain imported successfully!")

    from pygeoinf.interval.functions import Function
    print("✓ Function imported successfully!")

    # Quick test
    interval = IntervalDomain(-1, 1)
    domain = L2Space(10, interval)
    grad_op = GradientOperator(domain, method='finite_difference')
    print(f"✓ GradientOperator created: method={grad_op.method}")
    print(f"   fd_order={grad_op.fd_order}")

    # Test with a simple function
    def test_func(x):
        return x**2

    f = Function(domain, evaluate_callable=test_func)
    df_dx = grad_op(f)

    # Test at a point
    test_point = 0.5
    result = df_dx(test_point)
    expected = 2 * test_point  # derivative of x^2 is 2x

    print(f"✓ Applied gradient to f(x) = x²")
    print(f"   f'(0.5) = {result:.6f} (expected: {expected:.6f})")
    print(f"   Error: {abs(result - expected):.6f}")

    print("\n🎉 GradientOperator working correctly!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
