import pygeoinf as inf
import matplotlib.pyplot as plt

from pygeoinf.symmetric_space.circle import Lebesgue, Sobolev


order = 2
scale = 0.01
HS = Sobolev.from_sobolev_parameters(order, scale)
L2 = Lebesgue(HS.kmax)


mu = HS.heat_kernel_gaussian_measure(0.1)
u = mu.sample()

points = [angle for angle in HS.angles() if not (1 < angle < 2)]

P = HS.point_evaluation_operator(points)

subspace = inf.LinearSubspace.from_kernel(P, solver=inf.MinResSolver())


v = subspace.project(u)


HS.plot(u)
HS.plot(v)

plt.show()
