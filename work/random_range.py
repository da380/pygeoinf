import matplotlib.pyplot as plt

# Import the core framework
import pygeoinf as inf
from pygeoinf.symmetric_space.circle import plot, Sobolev

# Import our new matrix-free module directly to avoid __init__ clashes


if __name__ == "__main__":
    # ==========================================
    # 1. Setup the Function Space and Operator
    # ==========================================
    kmax = 256
    radius = 1.0

    print("Initializing space and operator...")
    domain = Sobolev(kmax, 2, 0.01, radius=radius)

    # A symmetric, positive-definite smoothing operator
    A = domain.invariant_automorphism(lambda k: (1 + 0.001 * k * k) ** -1.0)

    # A structured prior measure to draw test vectors from
    measure = domain.heat_kernel_gaussian_measure(0.001)

    # ==========================================
    # 3. Test the Factorizations
    # ==========================================
    print("\nComputing abstract factorizations...")

    # SVD
    A_svd = inf.LowRankSVD.from_randomized(A, 10, max_rank=50, measure=measure)

    # Eigendecomposition (Valid because A is self-adjoint)
    A_eig = inf.LowRankEig.from_randomized(A, 10, max_rank=50, measure=measure)

    # Cholesky (Valid because A is positive-definite)
    A_chol = inf.LowRankCholesky.from_randomized(A, 10, max_rank=50, measure=measure)

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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    plot(domain, u_test, ax=ax1, label="u", color="black", linewidth=3)

    # Plot true continuous application with a thick line
    plot(domain, y_true, ax=ax2, label="True A(u)", color="black", linewidth=3)

    # Plot approximations with different dashed styles
    plot(domain, y_svd, ax=ax2, label="SVD Approx", linestyle="--", color="tab:blue")
    plot(domain, y_eig, ax=ax2, label="Eig Approx", linestyle="-.", color="tab:orange")
    plot(
        domain,
        y_chol,
        ax=ax2,
        label="Chol Approx",
        linestyle=":",
        color="tab:green",
        linewidth=2,
    )

    ax2.legend()
    plt.tight_layout()
    plt.show()
