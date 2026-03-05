import numpy as np
import matplotlib.pyplot as plt
from pygeoinf.symmetric_space_new.sphere import Lebesgue, Sobolev

X = Sobolev(64, 2, 0.1, radius=10)

mu = X.point_value_scaled_heat_kernel_gaussian_measure(0.1 * X.radius)

u = mu.sample_pointwise_std(1000)

fig, ax, im = X.plot(u)
fig.colorbar(im)
plt.show()
