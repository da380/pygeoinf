import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from pygeoinf.geometry.interval import Sobolev
from pygeoinf import LinearForm

X = Sobolev(0, pi, 0.001, 2, 0.05)


f = lambda x: np.exp(-10 * (x - pi / 2) ** 2)


u = X.project_function(f)
X.plot(u)

A = X.derivative_operator()

v = A(u)

X.plot(v)

plt.show()
