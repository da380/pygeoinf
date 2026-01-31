import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from pygeoinf import EuclideanSpace, LinearOperator
from pygeoinf.convex_optimisation import SubgradientDescent
from pygeoinf.convex_analysis import BallSupportFunction
from pygeoinf.backus_gilbert import DualMasterCostFunction
matplotlib.use('TkAgg')  # Use a non-interactive backend for environments without display

D = EuclideanSpace(dim=1)
M = EuclideanSpace(dim=1)
P = EuclideanSpace(dim=1)
G = LinearOperator(M, D, lambda m: 1.0 * m)
T = LinearOperator(M, P, lambda m: m)

d_tilde = np.array([1.0])
data_error_support = BallSupportFunction(D, center=np.array([1.0]), radius=0.5)
model_prior_support = BallSupportFunction(M, center=np.array([0.0]), radius=1.0)

q_direction = P.basis_vector(0)
cost_function = DualMasterCostFunction(
    data_space=D,
    property_space=P,
    model_space=M,
    G=G,
    T=T,
    model_prior_support=model_prior_support,
    data_error_support=data_error_support,
    observed_data=d_tilde,
    q_direction=q_direction,
)

solver = SubgradientDescent(cost_function, step_size=0.1, store_iterates=True)

print("Solving for x0 = [2.0]")
result = solver.solve(np.array([2.0]))
print("  Optimal point:", result.x_final)
print("  Optimal value:", result.f_final)

# Plot cost function and solver iterates
x_grid = np.linspace(-6.0, 3.0, 400)
f_grid = np.array([cost_function(np.array([x])) for x in x_grid])

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(x_grid, f_grid, label="φ(λ; q) (dual master cost)")

if result.iterates:
    iterates = np.array(result.iterates).reshape(-1)
    f_iters = np.array([cost_function(np.array([x])) for x in iterates])
    ax.plot(iterates, f_iters, "o-", ms=3, label="Subgradient steps")

ax.axhline(
    np.min(f_grid),
    color="gray",
    linestyle="--",
    linewidth=1,
    label="min φ(λ; q)",
)
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title("Subgradient Descent: Cost Function and Iterates")
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()