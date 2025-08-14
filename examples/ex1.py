import numpy as np
import matplotlib.pyplot as plt
import pygeoinf as inf
from pygeoinf.symmetric_space.sphere import Sobolev

X = Sobolev(128, 2, 0.1)


mu = X.sobolev_gaussian_measure(2, 0.3, 1)

x = mu.sample()

fig, ax, im = X.plot(x)

fig.colorbar(
    im, ax=ax, orientation="horizontal", shrink=0.7, label="Present-day ice thickness"
)

plt.show()
