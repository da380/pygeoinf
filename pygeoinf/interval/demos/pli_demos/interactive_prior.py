"""
Interactive Prior Visualization

This script provides an interactive GUI for exploring Gaussian prior measures
on function spaces with adjustable parameters for Sobolev regularity, length scale,
and overall variance. Users can define a true model and adjust hyperparameters
to ensure the prior captures the true model's behavior.

Usage:
    python interactive_prior.py
"""
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plots
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
import seaborn as sns
import os

from pygeoinf.interval.interval_domain import IntervalDomain
from pygeoinf.interval import Lebesgue
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
    inner_product=IntegrationConfig(method='simpson', n_points=1000),
    dual=IntegrationConfig(method='simpson', n_points=1000),
    general=IntegrationConfig(method='simpson', n_points=1000)
)

laplacian_integration_cfg = IntegrationConfig(
    method='simpson',
    n_points=1000
)

parallel_cfg = ParallelConfig(
    enabled=True,
    n_jobs=8
)

# Create model space
print("Creating model space...")
function_domain = IntervalDomain(0, 1, boundary_type='open', open_epsilon=1e-12)
N = 100  # dimension
M = Lebesgue(
    N,
    function_domain,
    basis='cosine',
    integration_config=Lebesgue_integration_cfg,
    parallel_config=parallel_cfg
)

# Fixed parameters
K = 100  # number of KL modes
x = function_domain.uniform_mesh(1000)
num_samples = 15
random_seed = 42

# Boundary condition options
bc_types = {
    'neumann': ('neumann', 0, 0),
    'dirichlet': ('dirichlet', 0, 0),
    'mixed_dn': ('mixed_neumann_dirichlet', 0, 0),  # Dirichlet left, Neumann right
    'mixed_nd': ('mixed_dirichlet_neumann', 1, 0),  # Neumann left, Dirichlet right
    'periodic': ('periodic', None, None),
}

# Initial boundary condition
current_bc = 'neumann'
bcs = BoundaryConditions(bc_type=bc_types[current_bc][0],
                        left=bc_types[current_bc][1],
                        right=bc_types[current_bc][2])

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
    'sine_linear': lambda x: np.exp(-((x - function_domain.center)/0.5)**2) * np.sin(5 * np.pi * x) + x,

}

# Initial selections
current_mean = 'linear'
current_true_model = 'smooth'
m_0 = Function(M, evaluate_callable=mean_functions[current_mean])
true_model = Function(M, evaluate_callable=true_model_functions[current_true_model]) if current_true_model != 'none' else None

print(f"Model space: {M.dim}-dimensional Lebesgue space on [{function_domain.a}, {function_domain.b}]")
print(f"KL modes: {K}")
print(f"Number of samples: {num_samples}")
print("\nStarting interactive visualization...")


def compute_prior(s, length_scale, overall_variance):
    """Compute prior measure with given parameters."""
    np.random.seed(random_seed)

    k = np.power(overall_variance, -0.5/s)
    alpha = (length_scale**2) * k**2

    L = Laplacian(M, bcs, alpha, method='spectral', dofs=100,
                  integration_config=laplacian_integration_cfg)
    C_0 = BesselSobolevInverse(M, M, k, s, L, dofs=100, n_samples=2048)
    sampler = KLSampler(C_0, mean=m_0, n_modes=K)
    M_prior = GaussianMeasure(covariance=C_0, expectation=m_0, sample=sampler.sample)

    return M_prior, C_0, k, alpha


def plot_prior(ax, M_prior, C_0):
    """Plot prior samples, mean, uncertainty bands, and true model."""
    ax.clear()

    # Draw samples
    for i in range(num_samples):
        sample = M_prior.sample()
        ax.plot(x, sample.evaluate(x), color='tab:blue', alpha=0.25,
                linewidth=1, label='Prior Samples' if i == 0 else "")

    # Compute pointwise standard deviation
    std_M = M.zero
    for i in range(K):
        eigenvalue = C_0.get_eigenvalue(i)
        eigenfunction = C_0.get_eigenfunction(i)
        std_M += eigenvalue * eigenfunction * eigenfunction

    std_values = np.sqrt(std_M.evaluate(x))
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
fig = plt.figure(figsize=(16, 8))
ax_main = plt.axes([0.1, 0.25, 0.65, 0.65])

# Initial parameters
init_s = 3.0
init_length_scale = 0.5
init_overall_variance = 0.1

# Compute and plot initial prior
print("Computing initial prior...")
M_prior, C_0, k, alpha = compute_prior(init_s, init_length_scale, init_overall_variance)
std_values = plot_prior(ax_main, M_prior, C_0)
ax_main.set_title(f"Prior Measure (s={init_s:.1f}, length_scale={init_length_scale:.2f}, variance={init_overall_variance:.3f})",
                  fontsize=16)

# Create sliders
ax_s = plt.axes([0.12, 0.15, 0.6, 0.02])
ax_length = plt.axes([0.12, 0.10, 0.6, 0.02])
ax_variance = plt.axes([0.12, 0.05, 0.6, 0.02])

# Create radio buttons for boundary conditions
ax_bc = plt.axes([0.78, 0.72, 0.18, 0.18])
radio_bc = RadioButtons(ax_bc, list(bc_types.keys()), active=list(bc_types.keys()).index(current_bc))
ax_bc.set_title('Boundary Conditions', fontsize=11, fontweight='bold')

# Create radio buttons for prior mean
ax_mean = plt.axes([0.78, 0.42, 0.18, 0.28])
radio_mean = RadioButtons(ax_mean, list(mean_functions.keys()), active=list(mean_functions.keys()).index(current_mean))
ax_mean.set_title('Prior Mean', fontsize=11, fontweight='bold')

# Create radio buttons for true model
ax_true = plt.axes([0.78, 0.05, 0.18, 0.35])
radio_true = RadioButtons(ax_true, list(true_model_functions.keys()), active=list(true_model_functions.keys()).index(current_true_model))
ax_true.set_title('True Model', fontsize=11, fontweight='bold')

slider_s = Slider(ax_s, 's (regularity)', 0.5, 6.0, valinit=init_s, valstep=0.5)
slider_length = Slider(ax_length, 'Length scale', 0.05, 2.0, valinit=init_length_scale, valstep=0.05)
slider_variance = Slider(ax_variance, 'log₁₀(Variance)', -5, 5, valinit=np.log10(init_overall_variance), valstep=0.1)


def update(val):
    """Update plot when sliders change."""
    s = slider_s.val
    length_scale = slider_length.val
    overall_variance = 10 ** slider_variance.val

    print(f"\nUpdating: s={s:.2f}, length_scale={length_scale:.2f}, variance={overall_variance:.3f}")

    M_prior, C_0, k, alpha = compute_prior(s, length_scale, overall_variance)
    std_values = plot_prior(ax_main, M_prior, C_0)

    ax_main.set_title(f"Prior Measure (s={s:.1f}, length_scale={length_scale:.2f}, variance={overall_variance:.3f})",
                      fontsize=16)

    print(f"  k={k:.4f}, α={alpha:.6f}")
    print(f"  Max std: {std_values.max():.4f}, Mean std: {std_values.mean():.4f}")

    fig.canvas.draw_idle()


def update_bc(label):
    """Update boundary conditions."""
    global bcs, current_bc
    current_bc = label
    bc_info = bc_types[label]
    bcs = BoundaryConditions(bc_type=bc_info[0], left=bc_info[1], right=bc_info[2])
    print(f"\nBoundary condition changed to: {label}")
    update(None)


def update_mean(label):
    """Update prior mean function."""
    global m_0, current_mean
    current_mean = label
    m_0 = Function(M, evaluate_callable=mean_functions[label])
    print(f"\nPrior mean changed to: {label}")
    update(None)


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
    s = slider_s.val
    length_scale = slider_length.val
    overall_variance = 10 ** slider_variance.val
    M_prior, C_0, k, alpha = compute_prior(s, length_scale, overall_variance)
    plot_prior(ax_main, M_prior, C_0)
    ax_main.set_title(f"Prior Measure (s={s:.1f}, length_scale={length_scale:.2f}, variance={overall_variance:.3f})",
                      fontsize=16)
    fig.canvas.draw_idle()


# Connect sliders and radio buttons
slider_s.on_changed(update)
slider_length.on_changed(update)
slider_variance.on_changed(update)
radio_bc.on_clicked(update_bc)
radio_mean.on_clicked(update_mean)
radio_true.on_clicked(update_true_model)

# Add instructions
fig.text(0.5, 0.01,
         'Use sliders for hyperparameters and radio buttons to change prior mean/true model. '
         'Close window to exit.',
         ha='center', fontsize=9, style='italic')

print("\n" + "="*60)
print("INTERACTIVE MODE")
print("="*60)
print("\nAdjust the sliders to explore different prior configurations:")
print("  • s (regularity): Controls smoothness (higher = smoother)")
print("  • Length scale: Controls spatial correlation (higher = more global)")
print("  • Variance: Controls amplitude (higher = larger fluctuations)")
print("\nClose the window when done.")
print("="*60 + "\n")

plt.show()
