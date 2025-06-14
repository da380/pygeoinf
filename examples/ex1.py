import numpy as np
from numpy import pi, sqrt, sin, cos
import matplotlib.pyplot as plt

from pygeoinf.homogeneous_space.circle import Sobolev


X = Sobolev.from_sobolev_parameters(1, 0.1, power_of_two=True)
print(X.dim)

u = X.dirac_representation(pi)

X.plot(u)
plt.show()
