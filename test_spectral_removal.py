import sys
sys.path.insert(0, '/home/adrian/PhD/Inferences/pygeoinf')

from pygeoinf.interval.operators import GradientOperator
from pygeoinf.interval.l2_space import L2Space
from pygeoinf.interval.interval_domain import IntervalDomain

def test_spectral_removal():
    """Test that spectral method is no longer available."""

    interval = IntervalDomain(-1, 1)
    domain = L2Space(10, interval)

    print("Testing that spectral method is removed...")

    try:
        grad_op = GradientOperator(domain, method='spectral')
        print("❌ ERROR: Spectral method should not be available!")
    except ValueError as e:
        print(f"✓ Correct: {e}")

    # Test that valid methods still work
    try:
        grad_op_fd = GradientOperator(domain, method='finite_difference')
        print("✓ Finite difference method still works")

        grad_op_auto = GradientOperator(domain, method='automatic')
        print("❌ ERROR: Should fail without JAX")
    except ImportError as e:
        print("✓ Automatic method properly requires JAX")

if __name__ == "__main__":
    test_spectral_removal()
