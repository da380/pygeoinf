import numpy as np
import matplotlib.pyplot as plt
from pygeoinf.symmetric_space_new.circle import Lebesgue, Sobolev

kmax = 256
order = 2
scale = 0.05

X = Sobolev(kmax, order, scale, radius=5)


mu = X.point_value_scaled_heat_kernel_gaussian_measure(0.5)

u = mu.sample_pointwise_variance(10000)

X.plot(u)

plt.show()
