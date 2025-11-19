"""
Interactive Prior Visualization - Sobolev Spaces

This script provides an interactive GUI for exploring Gaussian prior measures
on Sobolev spaces with adjustable parameters for both the Sobolev space structure
and the covariance operator. Users can define a true model and adjust hyperparameters
to ensure the prior captures the true model's behavior.

Usage:
    python interactive_prior_sobolev.py
"""
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plots
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
import seaborn as sns
import os

from pygeoinf.interval.interval_domain import IntervalDomain
from pygeoinf.interval import Lebesgue, Sobolev
from pygeoinf.interval.configs import IntegrationConfig, ParallelConfig, LebesgueIntegrationConfig
from pygeoinf.interval.functions import Function
from pygeoinf.gaussian_measure import GaussianMeasure
from pygeoinf.interval.boundary_conditions import BoundaryConditions
from pygeoinf.interval.operators import BesselSobolevInverse, Laplacian
from pygeoinf.interval.KL_sampler import KLSampler


# Set up folder for saving figures
figures_folder = 'prior_figures'
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder)

# Set plot style
sns.set_theme(style="whitegrid", palette="muted", color_codes=True)

# Configuration
print("Setting up configuration...")
Lebesgue_integration_cfg = LebesgueIntegrationConfig(
    inner_product=IntegrationConfig(method='simpson', n_points=500),
    dual=IntegrationConfig(method='simpson', n_points=500),
    general=IntegrationConfig(method='simpson', n_points=500)
)

laplacian_integration_cfg = IntegrationConfig(
    method='simpson',
    n_points=500
)

parallel_cfg = ParallelConfig(
    enabled=True,
    n_jobs=8
)

# Create function domain
print("Creating function domain...")
function_domain = IntervalDomain(0, 1, boundary_type='open', open_epsilon=1e-12)
N = 100  # dimension

# Fixed parameters
K = 50  # number of KL modes
x = function_domain.uniform_mesh(500)
num_samples = 10
random_seed = 42

# Boundary condition options
bc_types = {
    'neumann': 'neumann',
    'dirichlet': 'dirichlet',
    'mixed_dirichlet_neumann': 'mixed_dirichlet_neumann',
    'mixed_neumann_dirichlet': 'mixed_neumann_dirichlet',
    'periodic': 'periodic',
}

# Initial boundary condition
current_bc = 'neumann'
bcs = BoundaryConditions(bc_type=bc_types[current_bc])

# Available prior mean functions
mean_functions = {
    'linear': lambda x: x,
    'constant': lambda x: np.zeros_like(x),
    'quadratic': lambda x: x**2,
    'sine': lambda x: np.sin(2*np.pi*x),
    'exp_decay': lambda x: np.exp(-5*x),
}

# Available true model functions
true_model_functions = {
    'none': None,
    'smooth': lambda x: 0.5 * np.sin(4*np.pi*x) + 0.2 * np.cos(8*np.pi*x),
    'rough': lambda x: 0.3 * np.sin(20*np.pi*x),
    'step': lambda x: np.where(x < 0.5, 0.2, 0.8),
    'peak': lambda x: np.exp(-50*(x-0.5)**2),
    'polynomial': lambda x: 2*x**3 - 3*x**2 + x,
}

# Initial selections
current_mean = 'linear'
current_true_model = 'smooth'

# Sobolev space hyperparameters
init_s_sobolev = 1.0
init_length_scale_sobolev = 0.5
init_overall_variance_sobolev = 1.0

# Prior covariance hyperparameters
init_s_cov = 1.0
init_length_scale_cov = 0.5
init_overall_variance_cov = 0.1

# Global variables
M = None
M_laplacian = None
m_0 = None
true_model = None
laplacian_cache = {}


def create_sobolev_space(s_sobolev, length_scale_sobolev, overall_variance_sobolev):
    """Create Sobolev space with given hyperparameters."""
    global M, M_laplacian, m_0, true_model

    # Create underlying Lebesgue space for Laplacian
    M_laplacian = Lebesgue(
        N, function_domain, basis='cosine',
        integration_config=Lebesgue_integration_cfg,
        parallel_config=parallel_cfg
    )

    # Create Laplacian for Sobolev space
    k_sobolev = np.power(overall_variance_sobolev, -0.5/s_sobolev)
    alpha_sobolev = (length_scale_sobolev**2) * k_sobolev**2
    L_sobolev = Laplacian(
        M_laplacian, bcs, alpha_sobolev, method='spectral',
        dofs=50, n_samples=1024,
        integration_config=laplacian_integration_cfg
    )

    # Create Sobolev space
    M = Sobolev(N, function_domain, s_sobolev, k_sobolev, L_sobolev, basis='cosine')

    print(f"Sobolev space: H^{s_sobolev} with k={k_sobolev:.4f}, α={alpha_sobolev:.6f}")

    # Recreate functions with new space
    m_0 = Function(M, evaluate_callable=mean_functions[current_mean])
    if current_true_model != 'none':
        true_model = Function(M, evaluate_callable=true_model_functions[current_true_model])
    else:
        true_model = None

    return M


# Initialize Sobolev space
M = create_sobolev_space(init_s_sobolev, init_length_scale_sobolev, init_overall_variance_sobolev)

print(f"KL modes: {K}")
print(f"Number of samples: {num_samples}")
print("\nStarting interactive visualization...")


def compute_prior(s_cov, length_scale_cov, overall_variance_cov):
    """Compute prior measure with given covariance parameters."""
    np.random.seed(random_seed)

    k_cov = np.power(overall_variance_cov, -0.5/s_cov)
    alpha_cov = (length_scale_cov**2) * k_cov**2

    # Use cache key for Laplacian to avoid recreating
    cache_key = (alpha_cov, current_bc)
    if cache_key in laplacian_cache:
        L = laplacian_cache[cache_key]
    else:
        # Laplacian must be built on underlying Lebesgue space
        L = Laplacian(M_laplacian, bcs, alpha_cov, method='spectral', dofs=50,
                      integration_config=laplacian_integration_cfg)
        laplacian_cache[cache_key] = L

    # Covariance operator on Sobolev space
    C_0 = BesselSobolevInverse(M, M, k_cov, s_cov, L, dofs=50, n_samples=1024)
    sampler = KLSampler(C_0, mean=m_0, n_modes=K)
    M_prior = GaussianMeasure(covariance=C_0, expectation=m_0, sample=sampler.sample)

    return M_prior, C_0, sampler, k_cov, alpha_cov


def plot_prior(ax, M_prior, sampler):
    """Plot prior samples, mean, uncertainty bands, and true model."""
    ax.clear()

    # Draw samples
    for i in range(num_samples):
        sample = M_prior.sample()
        ax.plot(x, sample.evaluate(x), color='tab:blue', alpha=0.25,
                linewidth=1, label='Prior Samples' if i == 0 else "")

    # Compute pointwise standard deviation using KLSampler
    variance_func = sampler.variance_function()
    std_values = np.sqrt(variance_func.evaluate(x))
    mean_values = M_prior.expectation.evaluate(x)

    # Plot mean and uncertainty
    ax.plot(x, mean_values, color='tab:green', linewidth=3,
            label='Prior Mean', zorder=10)
    ax.fill_between(x, mean_values - 2*std_values, mean_values + 2*std_values,
                     color='tab:blue', alpha=0.15, label='±2σ Band')
    ax.plot(x, mean_values + 2*std_values, color='tab:blue', linestyle='--',
            alpha=0.7, linewidth=1.5, label='±2σ Boundaries')
    ax.plot(x, mean_values - 2*std_values, color='tab:blue', linestyle='--',
            alpha=0.7, linewidth=1.5)

    # Plot true model if defined
    if true_model is not None:
        ax.plot(x, true_model.evaluate(x), color='tab:red', linewidth=2.5,
                linestyle='-', label='True Model', zorder=11)

    ax.set_xlabel(r"$x$", fontsize=14)
    ax.set_ylabel("Model Value", fontsize=14)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, linestyle=':', alpha=0.4)
    sns.despine(ax=ax)

    return std_values


# Create figure and axes
fig = plt.figure(figsize=(18, 9))
ax_main = plt.axes([0.08, 0.35, 0.60, 0.58])

# Compute and plot initial prior
print("Computing initial prior...")
M_prior, C_0, sampler, k, alpha = compute_prior(init_s_cov, init_length_scale_cov, init_overall_variance_cov)
std_values = plot_prior(ax_main, M_prior, sampler)
ax_main.set_title(f"Prior on Sobolev H^{init_s_sobolev:.1f} | Cov: s={init_s_cov:.1f}, ls={init_length_scale_cov:.2f}, var={init_overall_variance_cov:.3f}",
                  fontsize=14)

# Prior covariance hyperparameter sliders
ax_s_cov = plt.axes([0.10, 0.26, 0.55, 0.02])
ax_length_cov = plt.axes([0.10, 0.22, 0.55, 0.02])
ax_variance_cov = plt.axes([0.10, 0.18, 0.55, 0.02])

# Sobolev space hyperparameter sliders
ax_s_sobolev = plt.axes([0.10, 0.11, 0.55, 0.02])
ax_length_sobolev = plt.axes([0.10, 0.07, 0.55, 0.02])
ax_variance_sobolev = plt.axes([0.10, 0.03, 0.55, 0.02])

# Radio buttons - Boundary Conditions
ax_bc = plt.axes([0.70, 0.62, 0.12, 0.16])
radio_bc = RadioButtons(ax_bc, list(bc_types.keys()), active=list(bc_types.keys()).index(current_bc))
ax_bc.set_title('Boundary Conds', fontsize=10, fontweight='bold')

# Radio buttons - Prior Mean
ax_mean = plt.axes([0.70, 0.42, 0.12, 0.18])
radio_mean = RadioButtons(ax_mean, list(mean_functions.keys()), active=list(mean_functions.keys()).index(current_mean))
ax_mean.set_title('Prior Mean', fontsize=10, fontweight='bold')

# Radio buttons - True Model
ax_true = plt.axes([0.70, 0.10, 0.12, 0.30])
radio_true = RadioButtons(ax_true, list(true_model_functions.keys()), active=list(true_model_functions.keys()).index(current_true_model))
ax_true.set_title('True Model', fontsize=10, fontweight='bold')

# Text labels
fig.text(0.08, 0.30, 'Prior Covariance Hyperparameters', fontsize=11, fontweight='bold')
fig.text(0.08, 0.14, 'Sobolev Space Hyperparameters', fontsize=11, fontweight='bold')

# Prior covariance sliders
slider_s_cov = Slider(ax_s_cov, 's_cov', 0.5, 3.0, valinit=init_s_cov, valstep=0.5)
slider_length_cov = Slider(ax_length_cov, 'log₁₀(ls_cov)', -2, 1, valinit=np.log10(init_length_scale_cov), valstep=0.05)
slider_variance_cov = Slider(ax_variance_cov, 'log₁₀(var_cov)', -3, 1, valinit=np.log10(init_overall_variance_cov), valstep=0.1)

# Sobolev space sliders
slider_s_sobolev = Slider(ax_s_sobolev, 's_sob', 0.5, 3.0, valinit=init_s_sobolev, valstep=0.5)
slider_length_sobolev = Slider(ax_length_sobolev, 'log₁₀(ls_sob)', -2, 1, valinit=np.log10(init_length_scale_sobolev), valstep=0.05)
slider_variance_sobolev = Slider(ax_variance_sobolev, 'log₁₀(var_sob)', -1, 1, valinit=np.log10(init_overall_variance_sobolev), valstep=0.1)


def update_cov(val):
    """Update plot when covariance sliders change."""
    global sampler
    s_cov = slider_s_cov.val
    length_scale_cov = 10 ** slider_length_cov.val
    overall_variance_cov = 10 ** slider_variance_cov.val

    s_sobolev = slider_s_sobolev.val

    print(f"\nUpdating covariance: s_cov={s_cov:.2f}, ls={length_scale_cov:.2e}, var={overall_variance_cov:.2e}")

    M_prior, C_0, sampler, k, alpha = compute_prior(s_cov, length_scale_cov, overall_variance_cov)
    std_values = plot_prior(ax_main, M_prior, sampler)

    title = f"Prior on Sobolev H^{s_sobolev:.1f} | Cov: s={s_cov:.1f}, ls={length_scale_cov:.2e}, var={overall_variance_cov:.2e}"
    ax_main.set_title(title, fontsize=13)

    print(f"  k={k:.4f}, α={alpha:.6f}")
    print(f"  Max std: {std_values.max():.4f}, Mean std: {std_values.mean():.4f}")

    fig.canvas.draw_idle()


def update_sobolev(val):
    """Update Sobolev space when space sliders change."""
    global laplacian_cache

    s_sobolev = slider_s_sobolev.val
    length_scale_sobolev = 10 ** slider_length_sobolev.val
    overall_variance_sobolev = 10 ** slider_variance_sobolev.val

    print(f"\nUpdating Sobolev space: s={s_sobolev:.2f}, ls={length_scale_sobolev:.2e}, var={overall_variance_sobolev:.2e}")

    # Clear cache when space changes
    laplacian_cache.clear()

    # Recreate Sobolev space
    create_sobolev_space(s_sobolev, length_scale_sobolev, overall_variance_sobolev)

    # Recompute and plot
    update_cov(None)


def update_bc(label):
    """Update boundary conditions."""
    global bcs, current_bc, laplacian_cache
    current_bc = label
    bcs = BoundaryConditions(bc_type=bc_types[label])
    # Clear cache when boundary conditions change
    laplacian_cache.clear()
    print(f"\nBoundary condition changed to: {label}")
    update_cov(None)


def update_mean(label):
    """Update prior mean function."""
    global m_0, current_mean
    current_mean = label
    m_0 = Function(M, evaluate_callable=mean_functions[label])
    print(f"\nPrior mean changed to: {label}")
    update_cov(None)


def update_true_model(label):
    """Update true model function."""
    global true_model, current_true_model
    current_true_model = label
    if label == 'none':
        true_model = None
        print("\nTrue model: None")
    else:
        true_model = Function(M, evaluate_callable=true_model_functions[label])
        print(f"\nTrue model changed to: {label}")

    # Replot without recomputing prior
    global sampler
    s_cov = slider_s_cov.val
    length_scale_cov = 10 ** slider_length_cov.val
    overall_variance_cov = 10 ** slider_variance_cov.val
    s_sobolev = slider_s_sobolev.val

    M_prior, C_0, sampler, k, alpha = compute_prior(s_cov, length_scale_cov, overall_variance_cov)
    plot_prior(ax_main, M_prior, sampler)
    title = f"Prior on Sobolev H^{s_sobolev:.1f} | Cov: s={s_cov:.1f}, ls={length_scale_cov:.2e}, var={overall_variance_cov:.2e}"
    ax_main.set_title(title, fontsize=13)
    fig.canvas.draw_idle()


# Connect sliders and radio buttons
slider_s_cov.on_changed(update_cov)
slider_length_cov.on_changed(update_cov)
slider_variance_cov.on_changed(update_cov)
slider_s_sobolev.on_changed(update_sobolev)
slider_length_sobolev.on_changed(update_sobolev)
slider_variance_sobolev.on_changed(update_sobolev)
radio_bc.on_clicked(update_bc)
radio_mean.on_clicked(update_mean)
radio_true.on_clicked(update_true_model)

# Add instructions
fig.text(0.5, 0.01,
         'Adjust sliders to explore Sobolev space structure and prior covariance. '
         'Top sliders: covariance operator. Bottom sliders: Sobolev space (affects mass operator damping). '
         'Close window to exit.',
         ha='center', fontsize=9, style='italic')

print("\n" + "="*60)
print("INTERACTIVE MODE - SOBOLEV SPACES")
print("="*60)
print("\nAdjust the sliders to explore different configurations:")
print("\nPrior Covariance (top):")
print("  • s_cov: Covariance regularity (higher = smoother samples)")
print("  • ls_cov: Covariance length scale (spatial correlation)")
print("  • var_cov: Covariance variance (amplitude)")
print("\nSobolev Space (bottom):")
print("  • s_sob: Space regularity (affects mass operator)")
print("  • ls_sob: Space length scale")
print("  • var_sob: Space variance")
print("\nKey insight: Both sets of parameters affect the prior!")
print("  - s_cov controls covariance eigenvalue decay")
print("  - s_sob controls mass operator damping (1/√μ_i)")
print("\nClose the window when done.")
print("="*60 + "\n")

plt.show()
