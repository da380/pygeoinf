import numpy as np
import matplotlib.pyplot as plt
import pygeoinf as inf
from pygeoinf.symmetric_space.sphere import Sobolev

X = Sobolev(128, 2, 0.1)


mu = X.sobolev_gaussian_measure(2, 0.3, 1)

a = 2
x = mu.sample()
y = mu.sample()

z = y.copy()


def axpy(a, x, y):
    y.data += a * x.data


axpy(a, x, y)

fig, ax, im = X.plot(y)

fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.7)

fig, ax, im = X.plot(z)

fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.7)

plt.show()
