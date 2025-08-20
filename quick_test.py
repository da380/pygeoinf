import sys
sys.path.insert(0, '/home/adrian/PhD/Inferences/pygeoinf')

try:
    from pygeoinf.interval.operators import GradientOperator
    print("✓ GradientOperator imported successfully!")

    from pygeoinf.interval.function_space import L2Space
    print("✓ L2Space imported successfully!")

    from pygeoinf.interval.functions import Function
    print("✓ Function imported successfully!")

    # Quick test
    domain = L2Space(10, (-1, 1))
    grad_op = GradientOperator(domain, method='finite_difference')
    print(f"✓ GradientOperator created: method={grad_op.method}, fd_order={grad_op.fd_order}")

    print("\n🎉 All imports and basic functionality work!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
