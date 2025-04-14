import numpy as np
from pygeoinf import linalg
import scipy.integrate as spi
from scipy.special import legendre
from sola.main_classes import functions, domains
from sola.aux import predefined_functions
import matplotlib.pyplot as plt
from line_profiler import LineProfiler
import sys

# Define the data and property spaces
D_dim = 5
P_dim = 1

D = linalg.EuclideanSpace(D_dim)
P = linalg.EuclideanSpace(P_dim)

# Define Model space

# Discretization of the model space
M_dim = 5

# Define the basis functions (Legendre polynomials on [0,1])
domain = domains.HyperParalelipiped([[0, 1]])
basis_functions_evaluation = [
    lambda x, n=n: legendre(n)(2 * x - 1)
    for n in range(M_dim)
]
basis_functions = [
    functions.Function(
        domain,
        evaluate_callable=basis_function
    )
    for basis_function in basis_functions_evaluation
]


# Define the inner product on the model space
def M_inner_product(f: functions.Function, g: functions.Function):
    return spi.quad(
        lambda x: f.evaluate(x) * g.evaluate(x),
        f.domain.bounds[0][0],
        f.domain.bounds[0][1]
    )[0]

# Convert function to coefficients (projection onto basis)
def M_to_components(f: functions.Function, basis_functions):
    return [
        M_inner_product(f, basis_function)
        for basis_function in basis_functions
    ]

# Convert coefficients back to function (linear combination)
def M_from_components(c, M_dim):
    function_evaluation = lambda x: sum(
        c[i] * basis_functions[i].evaluate(x)
        for i in range(M_dim)
    )
    return functions.Function(
        domain,
        evaluate_callable=function_evaluation
    )

M = linalg.HilbertSpace(M_dim, lambda f: M_to_components(f, basis_functions),
                        lambda c: M_from_components(c, M_dim),
                        M_inner_product)

# Define G mapping
kernels = [
    predefined_functions.NormalModes_1D(
        domain, order=10, spread=0.1, max_freq=10
    ) for _ in range(D_dim)
]


# Define the G mapping function
# This function maps a function `f` in the model space to a vector in the data space
# by computing the inner product of `f` with each kernel in the `kernels` list.
def G_mapping(f: functions.Function, kernels):
    return np.array([M.inner_product(f, kernel) for kernel in kernels])

# Define G mapping
targets = [
    predefined_functions.Gaussian_1D(domain, center=0.5, width=0.1)
    for i in range(P_dim)
]

# Define the T mapping function
# This function maps a function `f` in the model space to a vector in the property space
# by computing the inner product of `f` with each target in the `targets` list.
def T_mapping(f: functions.Function, targets):
    return np.array([M.inner_product(f, target) for target in targets])

# Define the linear operator T
# This operator maps from the model space (M) to the property space (P) using the T_mapping function.
T = linalg.LinearOperator(M, P, lambda f: T_mapping(f, targets))

# Fake Model
m_true = predefined_functions.Random_1D(domain, seed=44)

################
# PROFILING AREA
################
# Create a LineProfiler object
profiler = LineProfiler()

# Add functions from linalg.py to the profiler
profiler.add_function(linalg.LinearOperator.__init__)
profiler.add_function(linalg.LinearOperator.__call__)
profiler.add_function(linalg.LinearOperator.__mapping)


# Enable the profiler
profiler.enable()

# Define the linear operator G
# This operator maps from the model space (M) to the data space (D) using the G_mapping function.
G = linalg.LinearOperator(M, D, lambda f: G_mapping(f, kernels))

# Fake data
d = G(m_true)

# Disable the profiler
profiler.disable()
"""
# Step 1: Compute GG^*
GG_star = G @ G.adjoint

# Step 2: Compute (GG^*)^-1 using a solver
solver = linalg.CGSolver(rtol=1e-8, atol=1e-10, maxiter=1000)  # Conjugate Gradient Solver
GG_star_inverse = solver(GG_star)

a = GG_star_inverse(d) """

# Save the profiling results to a file
with open('linalg_profile_results.txt', 'w') as f:
    sys.stdout = f
    profiler.print_stats(output_unit=1e-6)  # Output in microseconds
    sys.stdout = sys.__stdout__