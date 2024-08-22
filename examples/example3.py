import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from pyshtools import SHGrid, SHCoeffs
from pyshtools.expand import MakeGridPoint


from linear_inference.vector_space import LinearForm
from linear_inference.S2 import HS, L2

lmax = 128
X = L2(lmax, grid = "GLQ", vectors_as_SHGrid = False)

u = X.random()

l, m = 20, 5
cp = np.zeros(X.dimension)
cp[X.spherical_harmonic_index(l,m)] = 1
up1 = LinearForm(X, components = cp)
#up2 = LinearForm(X, mapping =  lambda ulm : ulm.coeffs[0,l,m])

v1 = X.from_dual(up1)
#v2 = X.from_dual(up2)

print(up1(u) - X.inner_product(v1, u))
#print(up2(u) - X.inner_product(v2, u))










