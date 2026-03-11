import matplotlib.pyplot as plt
from pygeoinf.symmetric_space.line import Sobolev, plot

X = Sobolev(256, 2, 0.05)

u = X.dirac_representation(0.5)

plot(X, u)

plt.show()
