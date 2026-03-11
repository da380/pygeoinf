import matplotlib.pyplot as plt
from pygeoinf.symmetric_space.line import Sobolev, plot

X = Sobolev(256, 2, 0.05)

u = X.project_function(lambda x: x)

plot(X, u, full=True)

plt.show()
