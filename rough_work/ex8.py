import numpy as np
import matplotlib.pyplot as plt
from pygeoinf.symmetric_space.sphere import Sobolev


# Set up the space
space = Sobolev(128, 2, 0.1)

# Set up the reference measure
mu = space.point_value_scaled_heat_kernel_gaussian_measure(0.1)

# Set up the function for scaling
f = space.project_function(lambda point: np.arctan(point[0] / 10))

# Set up the new measure


fig, ax, im = space.plot(f)
fig.colorbar(im, location="bottom", shrink=0.8)
plt.show()
