import numpy as np
import matplotlib.pyplot as plt
from pygeoinf.symmetric_space_new.circle import Lebesgue, Sobolev

kmax = 256
order = 2
scale = 0.05
X = Sobolev(kmax, order, scale)

u = X.dirac_representation(np.deg2rad(90))

X.plot(u)

plt.show()
