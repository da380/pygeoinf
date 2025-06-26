import numpy as np
import matplotlib.pyplot as plt
import pygeoinf as inf
from pygeoinf.symmetric_space.sphere import Sobolev


X = Sobolev(128, 2, 0.1)


mu = X.heat_gaussian_measure(0.5, 1)

u = mu.sample()

fig, ax, im = X.plot(u)
fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.7)

plt.show()
