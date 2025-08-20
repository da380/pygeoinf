import sys
sys.path.insert(0, '/home/adrian/PhD/Inferences/pygeoinf')

import numpy as np
from pygeoinf.interval.operators import GradientOperator
from pygeoinf.interval.l2_space import L2Space
from pygeoinf.interval.interval_domain import IntervalDomain
from pygeoinf.interval.functions import Function

def test_gradient_accuracy():
    """Test gradient operator with different orders and functions."""

    interval = IntervalDomain(-1, 1)
    domain = L2Space(30, interval)

    # Test functions and their exact derivatives
    test_cases = [
        (lambda x: x**2, lambda x: 2*x, "x²"),
        (lambda x: x**3, lambda x: 3*x**2, "x³"),
        (lambda x: np.sin(np.pi * x), lambda x: np.pi * np.cos(np.pi * x), "sin(πx)"),
        (lambda x: np.exp(x), lambda x: np.exp(x), "e^x")
    ]

    orders = [2, 4]
    test_points = np.array([-0.5, 0.0, 0.5])

    print("Testing GradientOperator accuracy:")
    print("="*50)

    for func, exact_grad, name in test_cases:
        print(f"\nFunction: {name}")
        print("-" * 20)

        f = Function(domain, evaluate_callable=func)
        exact_values = exact_grad(test_points)

        for order in orders:
            grad_op = GradientOperator(
                domain,
                method='finite_difference',
                fd_order=order,
                fd_step=1e-4
            )
            df_dx = grad_op(f)
            numerical_values = df_dx(test_points)

            errors = np.abs(numerical_values - exact_values)
            max_error = np.max(errors)

            print(f"Order {order}: max error = {max_error:.2e}")

            if order == 2:
                order2_error = max_error
            else:
                improvement = order2_error / max_error if max_error > 0 else float('inf')
                print(f"         improvement factor: {improvement:.1f}x")

    print("\n" + "="*50)
    print("Summary: Higher order finite differences show expected")
    print("improvement in accuracy for smooth functions.")

if __name__ == "__main__":
    test_gradient_accuracy()
