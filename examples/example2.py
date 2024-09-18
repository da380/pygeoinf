from pygeoinf.sphere import SHToolsHelper
import pyshtools as sh
import matplotlib.pyplot as plt
from pygeoinf.linalg import VectorSpace

lmax = 128


shtools_helper = SHToolsHelper(lmax)


X = VectorSpace(shtools_helper.dim, shtools_helper.to_components_from_SHGrid, shtools_helper.from_components_to_SHGrid)

u = X.random()

plt.pcolor(u.data)
plt.show()