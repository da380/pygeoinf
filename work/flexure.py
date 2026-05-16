import matplotlib.pyplot as plt
import pygeoinf as inf
import pygeoinf.symmetric_space.sphere as sph


# ===================================================================== #
#                           Main Execution                              #
# ===================================================================== #

if __name__ == "__main__":

    print("1. Initializing Spherical Space...")
    X = sph.Lebesgue(128)

    # Setup Physical Parameters
    nu = 0.25  # Poisson's ratio
    rho_g = 1.0  # Normalized restoring force (e.g., mantle/water density contrast)
    D0 = 0.0001  # Baseline oceanic flexural rigidity

    print("2. Constructing Spatially Varying Rigidity Field...")
    # Create base uniform rigidity
    D_base = X.project_function(lambda _: D0)

    # Stiffen continents by a factor of 11 using the domain mask
    D_raw = D_base * (1.0 + 4 * X.domain_mask())

    # Smooth the sharp coastlines using a heat kernel to prevent Gibbs ringing
    S = X.heat_kernel_gaussian_measure(0.05).covariance
    D = S(D_raw)

    print("3. Building Exact Forward Operator and Preconditioner...")
    A = X.thin_elastic_shell_operator(D, nu, rho_g)
    P = X.invariant_automorphism(lambda k: 1 / (D0 * k**2 + rho_g))

    print("4. Generating Random Geological Load...")
    # Use a smoothed random field to represent seamounts, ice sheets, etc.
    mu = X.heat_kernel_gaussian_measure(0.1)
    u = mu.sample()

    print("5. Solving the Flexure Equations...")
    # Baseline solve if the Earth were entirely oceanic (instantaneous)
    v0 = P(u)

    # Iterative CG solve for the exact variable rigidity
    solver = inf.CGSolver(callback=inf.ProgressCallback())
    v = solver(A, preconditioner=P)(u)

    print("6. Plotting Results...")

    # Plot 1: Rigidity map (D)
    ax, im = sph.plot(D, colorbar=True, cmap="viridis")
    ax.set_title("Smoothed Flexural Rigidity ($D$)")

    # Plot 2: Load (u)
    ax, im = sph.plot(u, colorbar=True, symmetric=True, cmap="RdBu_r")
    ax.set_title("Load ($q$)")

    # Plot 3: Baseline Flexure (Uniform Oceanic)
    ax, im = sph.plot(v0, colorbar=True, symmetric=True, cmap="RdBu_r")
    ax.set_title("Flexure (Uniform Oceanic Rigidity)")

    # Plot 4: True Flexure (Variable Rigidity)
    ax, im = sph.plot(v, colorbar=True, symmetric=True, cmap="RdBu_r")
    ax.set_title("True Flexure (Variable Continental Rigidity)")

    # Plot 5: Residual / Continent effect
    ax, im = sph.plot(v - v0, colorbar=True, symmetric=True, cmap="PRGn")
    ax.set_title("Effect of Continents ($v_{true} - v_{uniform}$)")

    plt.show()
