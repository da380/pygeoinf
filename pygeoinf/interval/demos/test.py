import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
import warnings
from collections import defaultdict
import os

# PyGeoInf imports
from pygeoinf.interval.function_providers import NormalModesProvider, BumpFunctionProvider
from pygeoinf.interval.interval_domain import IntervalDomain
from pygeoinf.interval.l2_space import L2Space
from pygeoinf.hilbert_space import EuclideanSpace, LinearOperator
from pygeoinf.interval.sola_operator import SOLAOperator
from pygeoinf.interval.l2_functions import Function
from pygeoinf.linear_solvers import CholeskySolver
from pygeoinf.gaussian_measure import GaussianMeasure

# Styling and output setup
sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams['figure.dpi'] = 100
warnings.filterwarnings('ignore', category=UserWarning)

# Create output directory
output_dir = 'sola_basis_benchmarks'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("SOLA Basis Function Benchmarking Study")
print("=" * 50)

# Fixed experimental parameters
DOMAIN = IntervalDomain(0, 1)
N_DATA = 50
N_PROPERTIES = 20
NOISE_LEVEL = 0.1
INTEGRATION_POINTS = 1000
RANDOM_SEED = 42

# Basis function types to test
BASIS_TYPES = ['sine', 'fourier']

# Number of basis functions to test
N_BASIS_VALUES = [5, 10, 15, 20, 30, 40, 50, 75, 100]

# Create spaces that don't change
D = EuclideanSpace(N_DATA)
P = EuclideanSpace(N_PROPERTIES)

# Target function parameters
width = 0.2
centers = np.linspace(DOMAIN.a + width / 2, DOMAIN.b - width / 2, N_PROPERTIES)

print(f"Testing {len(BASIS_TYPES)} basis types with {len(N_BASIS_VALUES)} different sizes")
print(f"Basis types: {BASIS_TYPES}")
print(f"N values: {N_BASIS_VALUES}")


######################################################################################################################################
def create_synthetic_model(M):
    """Create a complex synthetic model for testing basis representation."""
    def model_function(x):
        # Complex model with multiple features:
        # 1. Gaussian envelope
        # 2. High-frequency oscillation
        # 3. Linear trend
        # 4. Localized bump
        envelope = np.exp(-((x - DOMAIN.center) / 0.4)**2)
        oscillation = np.sin(5 * np.pi * x)
        trend = x
        bump = np.exp(-((x - 0.7) / 0.1)**2)
        return envelope * oscillation + 0.3 * trend + 0.5 * bump

    return Function(M, evaluate_callable=model_function)


def generate_noisy_data(G, m_bar, noise_level=NOISE_LEVEL, seed=RANDOM_SEED):
    """Generate noisy synthetic observations."""
    np.random.seed(seed)
    d_clean = G(m_bar)
    noise_std = noise_level * np.max(np.abs(d_clean))
    noise = np.random.normal(0, noise_std, d_clean.shape)
    return d_clean + noise, noise_std**2


# Visualization grid
x_viz = np.linspace(DOMAIN.a, DOMAIN.b, 1000)

print("Synthetic model and data generation functions defined.")

######################################################################################################################################
basis_type = 'fourier'
n_basis = 20
print(f"  Testing {basis_type} basis with N={n_basis}...")

# Create model space
M = L2Space(n_basis, DOMAIN, basis_type=basis_type)

# Create synthetic model
m_bar = create_synthetic_model(M)

# Time: Forward operator construction
t_start = time.time()
normal_modes_provider = NormalModesProvider(
    M, gaussian_width_percent_range=(5, 10),
    freq_range=(5, 10), random_state=39
)
G = SOLAOperator(M, D, normal_modes_provider,
                integration_method='trapz', n_points=INTEGRATION_POINTS)
t_forward = time.time() - t_start

# Time: Target operator construction
t_start = time.time()
target_provider = BumpFunctionProvider(M, centers=centers, default_width=width)
T = SOLAOperator(M, P, target_provider,
                integration_method='trapz', n_points=INTEGRATION_POINTS)
t_target = time.time() - t_start

# Generate data
d_noisy, noise_var = generate_noisy_data(G, m_bar)

# Time: Gram matrix computation
t_start = time.time()
Lambda = G @ G.adjoint
t_gram = time.time() - t_start


# Time: Noise covariance and solver setup (detailed timing)
t_start = time.time()
C_D_matrix = noise_var * np.eye(N_DATA)
t_C_D_matrix = time.time() - t_start

t_start = time.time()
gaussian_D = GaussianMeasure.from_covariance_matrix(D, C_D_matrix, expectation=d_noisy)
t_gaussian_D = time.time() - t_start

t_start = time.time()
cholesky_solver = CholeskySolver(galerkin=True)
t_cholesky_solver = time.time() - t_start

t_start = time.time()
W_inv = cholesky_solver(Lambda + gaussian_D.covariance)
t_W_inv = time.time() - t_start

# Time: SOLA operator construction
t_start = time.time()
G_inv = G.adjoint @ W_inv
X = T @ G_inv
t_sola = time.time() - t_start

# Time: Property inference
t_start = time.time()
property_measure = gaussian_D.affine_mapping(operator=X)
t_inference = time.time() - t_start

# Compute accuracy metrics
m_reconstructed = G_inv(d_noisy)
reconstruction_error = M.norm(m_reconstructed - m_bar) / M.norm(m_bar)

p_true = T(m_bar)
p_estimated = property_measure.expectation
property_error = np.mean(np.abs(p_estimated - p_true))

# Condition number analysis
gram_matrix = Lambda.matrix(dense=True)
condition_number = np.linalg.cond(gram_matrix)

# Print L2Space component call counts after Cholesky solve
print(f"L2Space._to_components calls: {M._to_components_count}")
print(f"L2Space._from_components calls: {M._from_components_count}")

# Store results with detailed timing
result = {
    'basis_type': basis_type,
    'n_basis': n_basis,
    'reconstruction_error': reconstruction_error,
    'property_error': property_error,
    'condition_number': condition_number,
    'time_forward': t_forward,
    'time_target': t_target,
    'time_gram': t_gram,
    'time_C_D_matrix': t_C_D_matrix,
    'time_gaussian_D': t_gaussian_D,
    'time_cholesky_solver': t_cholesky_solver,
    'time_W_inv': t_W_inv,
    'time_sola': t_sola,
    'time_inference': t_inference,
    'total_time': t_forward + t_target + t_gram + t_C_D_matrix + t_gaussian_D + t_cholesky_solver + t_W_inv + t_sola + t_inference
}


print(f"Results for {basis_type} basis with N={n_basis}:")
print(f"  Reconstruction error: {reconstruction_error:.4f}")
print(f"  Property error: {property_error:.4f}")
print(f"  Condition number: {condition_number:.4f}")
print("  Times (s):")
print(f"    Forward operator: {t_forward:.4f}")
print(f"    Target operator: {t_target:.4f}")
print(f"    Gram matrix: {t_gram:.4f}")
print(f"    C_D_matrix: {t_C_D_matrix:.6f}")
print(f"    GaussianMeasure: {t_gaussian_D:.6f}")
print(f"    CholeskySolver creation: {t_cholesky_solver:.6f}")
print(f"    W_inv (Cholesky solve): {t_W_inv:.6f}")
print(f"    SOLA operator: {t_sola:.4f}")
print(f"    Property inference: {t_inference:.4f}")
print(f"    Total: {result['total_time']:.4f}")