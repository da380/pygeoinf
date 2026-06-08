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

    # Stiffen continents by some factor
    D_raw = D_base * (1.0 + 9 * X.domain_mask())

    # Smooth the sharp coastlines using a heat kernel to prevent Gibbs ringing
    S = X.heat_kernel_gaussian_measure(0.1).covariance
    D = S(D_raw)

    P = X.invariant_automorphism(lambda k: 1 / (D0 * k**2 + rho_g))
    B = X.inverse_flexural_operator(
        D,
        nu,
        rho_g,
        baseline_rigidity=D0,
        solver=inf.CGSolver(callback=inf.ProgressCallback()),
    )

    print("4. Generating Random  Load...")
    # Use a smoothed random field
    mu = X.sobolev_kernel_gaussian_measure(2.0, 0.1)
    u = mu.sample()

    print("5. Solving the Flexure Equations...")
    # Baseline solve if the Earth were entirely oceanic (instantaneous)
    v0 = P(u)
    v = B(u)

    print("6. Plotting Results...")

    # Plot 1: Rigidity map (D)
    # We pass in high-level gridline intervals, but style the fonts via the attached object
    ax, im = sph.plot(
        D,
        colorbar=True,
        cmap="viridis",
        gridlines=True,
        gridlines_kwargs={"lat_interval": 45, "lon_interval": 60},
        coasts=True,  # Let's see the continents!
    )
    ax.set_title("Smoothed Flexural Rigidity ($D$)", pad=15)

    # Intercept the attached colorbar to adjust labels and ticks
    im.colorbar.set_label("Rigidity Value (normalized)", fontsize=12, fontweight="bold")
    im.colorbar.ax.tick_params(labelsize=10, color="gray")

    # Intercept the attached gridliner to style the latitude/longitude text
    ax.gridliner.xlabel_style = {"size": 10, "color": "gray", "weight": "bold"}
    ax.gridliner.ylabel_style = {"size": 10, "color": "gray", "weight": "bold"}

    # Plot 2: Load (u)
    ax, im = sph.plot(u, colorbar=True, symmetric=True, cmap="RdBu_r", coasts=True)
    ax.set_title("Load ($q$)", pad=15)
    im.colorbar.set_label("Load Amplitude", fontsize=10)

    # Plot 3: Baseline Flexure (Uniform Oceanic)
    ax, im = sph.plot(v0, colorbar=True, symmetric=True, cmap="RdBu_r", coasts=True)
    ax.set_title("Flexure (Uniform Oceanic Rigidity)", pad=15)
    im.colorbar.set_label("Deflection", fontsize=10)

    # Plot 4: True Flexure (Variable Rigidity)
    ax, im = sph.plot(v, colorbar=True, symmetric=True, cmap="RdBu_r", coasts=True)
    ax.set_title("True Flexure (Variable Continental Rigidity)", pad=15)
    im.colorbar.set_label("Deflection", fontsize=10)

    # Plot 5: Residual / Continent effect
    ax, im = sph.plot(
        v - v0, colorbar=True, symmetric=True, cmap="PRGn", gridlines=True, coasts=True
    )
    ax.set_title("Effect of Continents ($v_{true} - v_{uniform}$)", pad=15)

    # --- SHOWCASING NEW FUNCTIONALITY ---
    im.colorbar.set_label("Deflection Difference", fontsize=12)
    # We can apply completely different styling to the gridlines on this specific plot
    ax.gridliner.xlabel_style = {"color": "darkgreen", "style": "italic"}
    ax.gridliner.ylabel_style = {"color": "darkgreen", "style": "italic"}

    plt.show()
