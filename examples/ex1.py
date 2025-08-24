import matplotlib.pyplot as plt

import numpy as np

import pygeoinf as inf
from pygeoinf.symmetric_space_new.sphere import Lebesgue

# X = Sobolev(16, 2, 0.1)
X = Lebesgue(32)

u = X.project_function(
    lambda point: np.sin(point[0] * np.pi / 180) * np.cos(point[1] * np.pi / 180)
)
X.plot(u)
plt.show()
