import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from pyshtools import SHGrid, SHCoeffs
from pyshtools.expand import MakeGridPoint

from linear_inference.vector_space import HilbertSpace
from linear_inference.linear_form import LinearForm
from linear_inference.linear_operator import LinearOperator
from linear_inference.sphere import Sobolev, Lebesgue

lmax = 64
X = Sobolev(lmax, 2, 0.1)

u = X.random()

cp = np.zeros(X.dimension)

cp[X.component_index(2,1)] = 1

up = LinearForm(X, components = cp)

print(up(u))

v = X.from_dual(up)

plt.pcolor(v.data)
plt.show()








