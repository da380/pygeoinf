import matplotlib.pyplot as plt
import numpy as np
from pygeoinf.homogeneous_space.sphere import Sobolev


X = Sobolev(64, 2, 0.1)

mu = X.sobolev_gaussian_measure(2, 0.1, 1)

Q = mu.covariance

us = Q.fixed_rank_random_range(10)

for u in us:
    print(X.norm(u))

X.plot(us[0])
plt.colorbar()
plt.show()
