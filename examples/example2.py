import numpy as np
import pyshtools as sh
import matplotlib.pyplot as plt
import pygeoinf.linalg as la
import pygeoinf.sphere as sphere


def maximum_degree(space):
    return int(np.sqrt(space.dim)-1)

def dirac(space, latitude, longitude, /, *, degrees=True):
    helper = sphere.SHToolsHelper(maximum_degree(space))
    if degrees:
        colatitude = 90-latitude
    else:
        colatitude = np.pi/2-latitude
    coeffs = sh.expand.spharm(lmax, colatitude, longitude, normalization="ortho", degrees=degrees)
    c = helper.to_components_from_coeffs(coeffs)
    return space.dual.from_components(c)


def dirac_representation(space, latitude, longitude, /, *, degrees=True):
    up = dirac(space, latitude, longitude, degrees=degrees)
    return space.from_dual(up)

lmax = 256

X = sphere.Sobolev(lmax, 2, 0.1)

u = X.dirac_representation(90, 180)

plt.pcolor(u.data)
plt.show()

