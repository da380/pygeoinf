import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags
from pygeoinf import variable_rank_random_range, random_eig, fixed_rank_random_range

n = 100
m = 100


A = np.zeros((n, n))
for i in range(m):
    x = np.random.randn(n, 1)
    A += (1 / (i + 1) ** 5) * x @ x.T


Q = variable_rank_random_range(A, 4, rtol=1e-3)

U, eval = random_eig(A, Q)

D = diags([eval], [0])
Di = diags([np.reciprocal(eval)], [0])

B = U @ D @ U.T
C = U @ Di @ U.T

plt.matshow(A - B)
plt.colorbar()

plt.matshow(A @ C)
plt.colorbar()


plt.show()
