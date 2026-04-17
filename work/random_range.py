import numpy as np
import matplotlib.pyplot as plt

# Import the core framework
import pygeoinf as inf
from pygeoinf.symmetric_space.circle import Lebesgue, plot, Sobolev

# Import our new matrix-free module directly to avoid __init__ clashes
from pygeoinf import random_operators as ro


if __name__ == "__main__":
    # ==========================================
    # 1. Setup the Function Space and Operator
    # ==========================================
    kmax = 128
    radius = 1.0

    print("Initializing space and operator...")
    domain = Sobolev(kmax, 2, 0.1, radius=radius)

    # A symmetric, positive-definite smoothing operator
    A = domain.invariant_automorphism(lambda k: (1 + 0.001 * k * k) ** -2)

    # A structured prior measure to draw test vectors from
    measure = domain.heat_kernel_gaussian_measure(0.01)

    # ==========================================
    # 2. Abstract Randomized Range Finding
    # ==========================================
    target_rank = 5
    print(f"\nEstimating abstract range (target rank: {target_rank})...")

    # We use the unified 'random_range' wrapper from our new module
    q_basis = ro.random_range(A, measure, target_rank, method="variable", power=0)
    print(f"Captured {len(q_basis)} basis vectors.")

    # ==========================================
    # 3. Test the Factorizations
    # ==========================================
    print("\nComputing abstract factorizations...")

    # SVD
    A_svd = ro.LowRankSVD.from_randomized(A, measure, 10)

    # Eigendecomposition (Valid because A is self-adjoint)
    A_eig = ro.LowRankEig.from_randomized(A, measure, 10)

    # Cholesky (Valid because A is positive-definite)
    A_chol = ro.LowRankCholesky.from_randomized(A, measure, 10)

    # ==========================================
    # 4. Visual Verification
    # ==========================================
    print("\nGenerating random test field and verifying reconstructions...")
    u_test = measure.sample()

    # Apply the true continuous operator
    y_true = A(u_test)

    # Apply our low-rank approximations
    y_svd = A_svd(u_test)
    y_eig = A_eig(u_test)
    y_chol = A_chol(u_test)

    # Calculate relative errors
    norm_true = domain.norm(y_true)
    err_svd = domain.norm(domain.subtract(y_true, y_svd)) / norm_true
    err_eig = domain.norm(domain.subtract(y_true, y_eig)) / norm_true
    err_chol = domain.norm(domain.subtract(y_true, y_chol)) / norm_true

    print(f"  SVD Error:      {err_svd:.2e}")
    print(f"  Eig Error:      {err_eig:.2e}")
    print(f"  Cholesky Error: {err_chol:.2e}")

    # Plotting to visually confirm they all sit perfectly on top of one another
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot true continuous application with a thick line
    plot(domain, y_true, ax=ax, label="True A(u)", color="black", linewidth=3)

    # Plot approximations with different dashed styles
    plot(domain, y_svd, ax=ax, label="SVD Approx", linestyle="--", color="tab:blue")
    plot(domain, y_eig, ax=ax, label="Eig Approx", linestyle="-.", color="tab:orange")
    plot(
        domain,
        y_chol,
        ax=ax,
        label="Chol Approx",
        linestyle=":",
        color="tab:green",
        linewidth=2,
    )

    ax.set_title("Verification of Abstract Randomized Factorizations")
    ax.legend()
    plt.tight_layout()
    plt.show()
