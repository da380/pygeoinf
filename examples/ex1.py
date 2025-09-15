import pygeoinf as inf
import numpy as np
import matplotlib.pyplot as plt


# Generate a diagonally domainant matrix of dimension n whose eigenvalues are also decreasing
n = 1000
A = np.random.randn(n, n)


for i in range(n):
    A[i, i] += i

A = A @ A.T

diags = np.diag(A)


approx_diags = inf.random_diagonal(A, 100, rtol=1e-4, max_samples=10 * n)


plt.plot(diags, approx_diags, "o")
plt.show()
