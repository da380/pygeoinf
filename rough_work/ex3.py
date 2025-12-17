import pygeoinf as inf
from numpy import pi
import matplotlib.pyplot as plt

from pygeoinf.symmetric_space.circle import Lebesgue, Sobolev


order = 4
scale = 0.1
# HS = Sobolev.from_sobolev_parameters(order, scale)
HS = Sobolev(2**14, order, scale)
L2 = Lebesgue(HS.kmax)


mu = HS.heat_kernel_gaussian_measure(0.01)
u = mu.sample()
# u = HS.project_function(lambda th: 1)

m = L2.project_function(lambda th: 3 < th < 5)


def projection_mapping(u):
    return m * u


P_L2 = inf.LinearOperator.self_adjoint(L2, projection_mapping)

P = inf.LinearOperator.from_formal_adjoint(HS, L2, P_L2)

alpha = 1e-2

solver = inf.CGSolver()
B = solver(P @ P.adjoint + alpha * L2.identity_operator())
C = P.adjoint @ B @ P

v = P(u)

w = B(v)

for _ in range(10):
    r = v - (P @ P.adjoint)(w)
    wc = B(r)
    w += wc


z = P.adjoint(w)

HS.plot(u)
HS.plot(z)
HS.plot(u - z)


plt.show()
