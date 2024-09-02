import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


from pyshtools import SHGrid, SHCoeffs
from pyshtools.expand import spharm
from pyshtools.expand import MakeGridPoint


from linear_inference.vector_space import LinearForm
from linear_inference.two_sphere import HS, L2

lmax = 128
X = HS(lmax, 2, 0.2, grid = "GLQ")



ulm  = SHCoeffs.from_array(spharm(lmax, 90, 60, normalization= "ortho"), normalization= "ortho")
cp = X._to_components_from_SHCoeffs(ulm)
up = LinearForm(X, components = cp)

up = LinearForm(X, mapping = lambda u : u.data[20,30])


u = X.from_dual(up)


plt.pcolor(u.data)
plt.show()














