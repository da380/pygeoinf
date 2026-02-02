import numpy as np
import matplotlib.pyplot as plt
import pygeoinf as inf
from pygeoinf.symmetric_space.circle import Sobolev

order = 1.9
scale = 0.1

X = Sobolev(1000, order, scale)


prior_order = 0.0
prior_scale = 0.1
mu = X.point_value_scaled_sobolev_kernel_gaussian_measure(prior_order, prior_scale)

# u = mu.two_point_covariance(3)
u = X.dirac_representation(3)

X.plot(u)
plt.show()
