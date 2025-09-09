import numpy as np
from scipy.optimize import minimize

# 1. Define the deterministic quadratic problem using only NumPy
dim = 10
coeffs = np.arange(1, dim + 1)
shifts = np.arange(1, dim + 1) # This is the known true minimum

def objective_function(x):
    """
    Calculates f(x) = sum(coeffs * (x - shifts)**2).
    This is a simple, convex quadratic bowl.
    """
    return np.sum(coeffs * (x - shifts)**2)

# 2. Set the initial guess
x0 = np.zeros(dim)

print("--- Testing SciPy's derivative-free optimizers directly ---")
print(f"Problem: Minimize sum( (i+1) * (x_i - (i+1))^2 )")
print(f"Initial guess (x0): {x0}")
print(f"True minimum:       {shifts}")

# --- Test Nelder-Mead ---
print("\n" + "="*40)
print("Solver: Nelder-Mead")
print("="*40)
# Note: Nelder-Mead uses xatol and fatol for absolute tolerance
nm_options = {"maxiter": 5000, "xatol": 1e-4, "fatol": 1e-6, "disp": True}
nm_result = minimize(
    objective_function,
    x0,
    method='Nelder-Mead',
    options=nm_options,
)
print("-" * 40)
print(f"Success: {nm_result.success}")
print(f"Message: {nm_result.message}")
print(f"Function evaluations: {nm_result.nfev}")
print(f"Found minimum (x):    {np.round(nm_result.x, 4)}")
print(f"Difference:           {np.round(nm_result.x - shifts, 4)}")


# --- Test Powell ---
print("\n" + "="*40)
print("Solver: Powell")
print("="*40)
# Note: Powell uses xtol and ftol for relative tolerance
powell_options = {"maxiter": 5000, "xtol": 1e-4, "ftol": 1e-6, "disp": True}
powell_result = minimize(
    objective_function,
    x0,
    method='Powell',
    options=powell_options,
)
print("-" * 40)
print(f"Success: {powell_result.success}")
print(f"Message: {powell_result.message}")
print(f"Function evaluations: {powell_result.nfev}")
print(f"Found minimum (x):    {np.round(powell_result.x, 4)}")
print(f"Difference:           {np.round(powell_result.x - shifts, 4)}")