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
    A = domain.invariant_automorphism(lambda k: (1 + 0.001 * k * k) ** -0.5)

    # A structured prior measure to draw test vectors from
    measure = domain.heat_kernel_gaussian_measure(0.001)

    # ==========================================
    # 3. Test the Factorizations
    # ==========================================
    print("\nComputing abstract factorizations...")

    # SVD
    A_svd1 = inf.LowRankSVD.from_randomized(A, 10, measure=measure, max_rank=50)
    A_svd2 = inf.LowRankSVD.from_randomized(A, 10, measure=None, max_rank=50)
    A_svd3 = inf.LowRankSVD.from_randomized(
        A, 10, measure=inf.white_noise_measure(domain), max_rank=50
    )

    print(A_svd1.rank)
    print(A_svd2.rank)
    print(A_svd3.rank)

    # ==========================================
    # 4. Visual Verification
    # ==========================================
    print("\nGenerating random test field and verifying reconstructions...")
    u_test = measure.sample()

    # Apply the true continuous operator
    y_true = A(u_test)

    # Apply our low-rank approximations
    y1_svd = A_svd1(u_test)
    y2_svd = A_svd2(u_test)
    y3_svd = A_svd3(u_test)

    # Plotting to visually confirm they all sit perfectly on top of one another
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    plot(domain, u_test, ax=ax1, label="u", color="black", linewidth=3)

    # Plot true continuous application with a thick line
    plot(domain, y_true, ax=ax2, label="True A(u)", color="black", linewidth=3)

    # Plot approximations with different dashed styles
    plot(domain, y1_svd, ax=ax2, label="SVD Approx 1", linestyle="--", color="tab:blue")
    plot(
        domain, y2_svd, ax=ax2, label="SVD Approx 2", linestyle="--", color="tab:green"
    )
    plot(domain, y3_svd, ax=ax2, label="SVD Approx 3", linestyle="--", color="tab:red")

    ax2.legend()
    plt.tight_layout()
    plt.show()
