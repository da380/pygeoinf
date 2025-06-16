import numpy as np
from numpy import pi, sqrt, sin, cos
import matplotlib.pyplot as plt


from pygeoinf.homogeneous_space.line import Sobolev

X = Sobolev(512, 2, 0.5, x1=10)


mu = X.heat_gaussian_measure(0.1, 1)


u = mu.sample()


X.plot(u)
plt.show()
