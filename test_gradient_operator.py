#!/usr/bin/env python3
"""
Test script for the new GradientOperator.

This script demonstrates the gradient operator with finite difference method
since JAX is not available in this environment.
"""

import numpy as np
import matplotlib.pyplot as plt
from pygeoinf.interval.function_space import L2Space
from pygeoinf.interval.functions import Function
from pygeoinf.interval.operators import GradientOperator


def main():
    print("Testing GradientOperator with finite differences...")

    # Create domain
    domain = L2Space(30, (-1, 1))

    # Test function: f(x) = x^2, so f'(x) = 2x
    def test_func(x):
        return x**2

    def exact_derivative(x):
        return 2*x

    # Create Function object
    f = Function(domain, evaluate_callable=test_func, name="x²")

    # Create gradient operator with finite differences
    grad_op = GradientOperator(domain, method='finite_difference', fd_order=2)

    # Apply gradient
    df_dx = grad_op(f)

    # Test at various points
    test_points = np.linspace(-0.9, 0.9, 20)
    numerical_grad = df_dx(test_points)
    exact_grad = exact_derivative(test_points)

    # Compute error
    error = np.abs(numerical_grad - exact_grad)
    max_error = np.max(error)
    mean_error = np.mean(error)

    print(f"Maximum error: {max_error:.6f}")
    print(f"Mean error: {mean_error:.6f}")

    # Test with higher order finite differences
    print("\nTesting with 4th order finite differences...")
    grad_op_4th = GradientOperator(domain, method='finite_difference', fd_order=4)
    df_dx_4th = grad_op_4th(f)
    numerical_grad_4th = df_dx_4th(test_points)
    error_4th = np.abs(numerical_grad_4th - exact_grad)
    max_error_4th = np.max(error_4th)
    mean_error_4th = np.mean(error_4th)

    print(f"4th order - Maximum error: {max_error_4th:.6f}")
    print(f"4th order - Mean error: {mean_error_4th:.6f}")

    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot function and derivatives
    x_plot = np.linspace(-0.9, 0.9, 100)
    ax1.plot(x_plot, test_func(x_plot), 'k-', label='f(x) = x²', linewidth=2)
    ax1.plot(x_plot, exact_derivative(x_plot), 'r-', label="f'(x) = 2x (exact)", linewidth=2)
    ax1.plot(test_points, numerical_grad, 'bo', label="f'(x) FD (2nd order)", markersize=4)
    ax1.plot(test_points, numerical_grad_4th, 'g^', label="f'(x) FD (4th order)", markersize=4)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Function and its Gradient')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot errors
    ax2.semilogy(test_points, error, 'bo-', label='2nd order FD error', markersize=4)
    ax2.semilogy(test_points, error_4th, 'g^-', label='4th order FD error', markersize=4)
    ax2.set_xlabel('x')
    ax2.set_ylabel('Absolute Error')
    ax2.set_title('Gradient Computation Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/adrian/PhD/Inferences/pygeoinf/gradient_operator_test.png',
                dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to gradient_operator_test.png")

    # Test properties
    print(f"\nGradient operator properties:")
    print(f"FD step size: {grad_op.fd_step}")
    print(f"FD order: {grad_op.fd_order}")
    print(f"Method: {grad_op.method}")
    print(f"Domain: {grad_op.domain}")
    print(f"Codomain: {grad_op.codomain}")

    print("\nGradientOperator test completed successfully!")


if __name__ == "__main__":
    main()
