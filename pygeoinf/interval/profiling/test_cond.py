from pygeoinf.interval.function_providers import NormalModesProvider, BumpFunctionProvider
from pygeoinf.interval.interval_domain import IntervalDomain
from pygeoinf.interval.lebesgue_space import Lebesgue
from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.interval.operators import SOLAOperator
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
# Create a function domain and spaces
function_domain = IntervalDomain(0, 1)

M = Lebesgue(0, function_domain, basis=None)  # model space

N_ds = np.linspace(10, 100, 10, dtype=int)  # data space dimensions
conds = np.zeros_like(N_ds)  # to store condition numbers
seeds = np.arange(0, 10, 1)
max_N_d = np.zeros_like(seeds)
for i, seed in enumerate(seeds):
    for N_d in N_ds:
        D = EuclideanSpace(N_d) # data space

        # Create a normal modes provider for the forward operator
        # and a bump function provider for the target operator
        # Note: The random_state is set to ensure reproducibility of results
        normal_modes_provider = NormalModesProvider(
            M,
            n_modes_range=(1, 50),
            coeff_range=(-5, 5),
            gaussian_width_percent_range=(1, 5),
            freq_range=(0.1, 20),
            random_state=seed,
        )
        G = SOLAOperator(M, D, normal_modes_provider)

        G_mat = (G@G.adjoint).matrix(dense=True, galerkin=False, parallel=True, n_jobs=8)

        cond = np.linalg.cond(G_mat)
        if cond > 1e16:
            max_N_d[i] = N_d
            print(f"Seed {seed}, Data dim {N_d}, Condition number: {cond}")
            break

plt.figure()
plt.scatter(seeds, max_N_d)
plt.yscale("log")
plt.xlabel("Random seed")
plt.ylabel("Data dimension")
plt.title("Data dimension of the forward operator vs random seed")
plt.grid()
plt.show()